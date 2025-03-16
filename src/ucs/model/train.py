from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from ucs.core.loggers import Loggers
from ucs.data.data_module import SegmentationDataModule
from ucs.data.transform import Augmentation
from ucs.model.lightning_model import SegformerFinetuner
from ucs.utils import (
    find_checkpoint,
    get_next_version,
    load_class_weights,
    save_class_weights,
)
from ucs.utils.callbacks import SaveModel
from ucs.utils.metrics import compute_class_weights


def prepare_paths(config, resume_version=None):
    """
    Prepare directory paths for logging and checkpoints.
    """
    logs_dir = Path(config.directories.logs)
    version = (
        resume_version
        if resume_version
        else get_next_version(logs_dir, config.loggers.project_name)
    )

    return {
        "version": version,
        "pretrained_dir": Path(config.directories.pretrained) / f"version_{version}",
        "logs_dir": Path(config.directories.logs),
        "checkpoint_dir": Path(config.directories.checkpoints) / f"version_{version}",
    }


def initialize_callbacks(pretrained_dir, checkpoint_dir, callbacks_config):
    """
    Initialize callbacks for model training.
    """
    return [
        SaveModel(
            pretrained_dir=pretrained_dir,
            dirpath=checkpoint_dir,
            filename="{epoch}-val_loss={val/loss:.2f}-val_iou={val/mean_iou:.2f}",
            monitor=callbacks_config.save_model_monitor,
            save_top_k=1,
            mode=callbacks_config.save_model_mode,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=callbacks_config.early_stop_monitor,
            patience=callbacks_config.early_stop_patience,
            mode=callbacks_config.early_stop_mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def initialize_trainer(max_epochs, callbacks, loggers):
    """
    Initialize PyTorch Lightning Trainer.
    """
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        logger=loggers,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )


def get_class_weights(config, data_module):
    """
    Computes class weights based on the selected weighting strategy.

    Returns:
        torch.Tensor or None: A tensor containing class weights, or None if no weighting is applied.
    """
    if config.training.weighting_strategy == "none":
        return None
    strategy = config.training.weighting_strategy
    class_weights_dir = config.directories.logs
    class_weights = load_class_weights(class_weights_dir, strategy)
    if class_weights is not None:
        return class_weights
    # Compute class weights based on the weighting strategy
    data_module.prepare_data()
    data_module.setup(stage="fit")

    class_weights = compute_class_weights(
        dataloader=data_module.train_dataloader(),
        num_classes=len(config.training.id2label),
        normalize=config.training.weighting_strategy,
    )
    save_class_weights(config.directories.logs, class_weights, strategy)
    return class_weights


def train(config, resume_version=None):
    """
    Train the Segformer model with the provided configuration.
    """
    # Prepare paths
    paths = prepare_paths(config, resume_version)

    _loggers = Loggers(config.loggers, paths["logs_dir"], paths["version"])
    loggers = _loggers.get_loggers_list()

    # Initialize callbacks for trainer
    callbacks = initialize_callbacks(
        paths["pretrained_dir"],
        paths["checkpoint_dir"],
        callbacks_config=config.callbacks,
    )
    # Initialize trainer
    trainer = initialize_trainer(config.training.max_epochs, callbacks, loggers)

    # Initialize data module
    data_module = SegmentationDataModule(
        config=config.dataset,
        transform=Augmentation(),
    )

    # compute class weights
    class_weight = get_class_weights(config, data_module)

    # Initialize fine tune model
    model = SegformerFinetuner(config=config.training, class_weights=class_weight)

    # Resolve checkpoint path if resuming from a checkpoint
    resume_checkpoint = (
        find_checkpoint(config.directories.checkpoints, resume_version)
        if resume_version
        else None
    )

    # Log the session
    _loggers.log_training_details(config)
    _loggers.watch(model)

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)
