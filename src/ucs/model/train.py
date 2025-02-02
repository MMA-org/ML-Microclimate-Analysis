from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

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
        f"version_{resume_version}" if resume_version else get_next_version(logs_dir)
    )
    return {
        "version": version,
        "pretrained_dir": Path(config.directories.pretrained) / version,
        "logs_dir": Path(config.directories.logs),
        "checkpoint_dir": Path(config.directories.checkpoints) / version,
    }


def initialize_logger(logs_dir, resume_version):
    """
    Initialize TensorBoard logger.
    """
    return TensorBoardLogger(
        save_dir=logs_dir,
        version=f"version_{resume_version}" if resume_version else None,
        default_hp_metric=False,
    )


def initialize_callbacks(pretrained_dir, checkpoint_dir, callbacks_config):
    """
    Initialize callbacks for model training.
    """
    return [
        SaveModel(
            pretrained_dir=pretrained_dir,
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}-{val_mean_iou:.2f}",
            monitor=callbacks_config.save_model_monitor,
            save_top_k=1,
            mode=callbacks_config.save_model_mode,
        ),
        EarlyStopping(
            monitor=callbacks_config.early_stop_monitor,
            patience=callbacks_config.early_stop_patience,
            mode=callbacks_config.early_stop_mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def initialize_trainer(max_epochs, callbacks, logger):
    """
    Initialize PyTorch Lightning Trainer.
    """
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        logger=logger,
        gradient_clip_val=1.0,
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

    class_weights_dir = config.directories.logs
    class_weights = load_class_weights(class_weights_dir)
    if class_weights is not None:
        return class_weights
    # Compute class weights based on the weighting strategy
    data_module.prepare_data()
    data_module.setup(stage="fit")

    class_weights = compute_class_weights(
        dataloader=data_module.train_dataloader(),
        num_classes=len(config.training.id2label),
        normalize=config.training.weighting_strategy,  # Pass the strategy here
        ignore_index=config.training.ignore_index,
    )
    save_class_weights(config.directories.logs, class_weights)
    return class_weights


def log_training_info(config, version, tensorboard_logger):
    """
    Logs training information into TensorBoard.

    Args:
        config: The training configuration.
        version: Experiment version.
        tensorboard_logger: TensorBoard logger from PyTorch Lightning.
    """
    version = version.split("_")[-1] if "_" in version else version
    log_text = f"""
    🚀 Start Training - Version {version} 
    📂 Dataset: {config.dataset.dataset_path}  

    🔹 Training Parameters:  
    • Model: {config.training.model_name}  
    • Learning Rate: {config.training.learning_rate}  
    • Weight Decay: {config.training.weight_decay}  
    • Batch Size: {config.dataset.batch_size}  
    • Max Epochs: {config.training.max_epochs}  

    🎯 Focal Loss Settings:  
    • Ignore Index: {config.training.ignore_index}  
    • Weighting Strategy: {config.training.weighting_strategy}  
    • Gamma: {config.training.gamma}  
    """
    print(log_text.strip(), "\n")
    tensorboard_logger.experiment.add_text("Training Info", log_text)


def train(config, resume_version=None):
    """
    Train the Segformer model with the provided configuration.
    """
    # Prepare paths
    paths = prepare_paths(config, resume_version)

    # Initialize TensorBoard logger
    logger = initialize_logger(paths["logs_dir"], resume_version)

    # Initialize data module
    data_module = SegmentationDataModule(
        config=config.dataset,
        transform=Augmentation(),
        model_name=config.dataset.model_name,
    )

    # compute class weights
    class_weight = get_class_weights(config, data_module)

    # Initialize fine tune model
    model = SegformerFinetuner(config=config.training, class_weights=class_weight)

    # Initialize callbacks for trainer
    callbacks = initialize_callbacks(
        paths["pretrained_dir"],
        paths["checkpoint_dir"],
        callbacks_config=config.callbacks,
    )
    # Initialize trainer
    trainer = initialize_trainer(config.training.max_epochs, callbacks, logger)

    # Resolve checkpoint path if resuming from a checkpoint
    resume_checkpoint = (
        find_checkpoint(config.directories.checkpoints, resume_version)
        if resume_version
        else None
    )

    # Log the session
    log_training_info(config, paths["version"], logger)

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)
