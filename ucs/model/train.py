from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from model.lightning_model import SegformerFinetuner
from data.data_module import SegmentationDataModule
from utils import get_next_version, find_checkpoint
from ucs.utils.callbacks import SaveModel
from data.transform import Augmentation
import torch


def prepare_paths(config, resume_version=None):
    """
    Prepare directory paths for logging and checkpoints.
    """
    logs_dir = Path(config.directories.logs)
    version = f"version_{resume_version}" if resume_version else get_next_version(
        logs_dir)
    return {
        "pretrained_dir": Path(config.directories.pretrained) / version,
        "logs_dir": logs_dir,
        "checkpoint_dir": Path(config.directories.checkpoints) / version,
    }


def initialize_logger(logs_dir, resume_version):
    """
    Initialize TensorBoard logger.
    """
    return TensorBoardLogger(
        save_dir=logs_dir,
        version=f"version_{resume_version}" if resume_version else None,
        default_hp_metric=False
    )


def initialize_callbacks(pretrained_dir, checkpoint_dir, callbacks_config):
    """
    Initialize callbacks for model training.
    """
    return [
        SaveModel(
            pretrained_dir=pretrained_dir,
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.2f}-{val_mean_iou:.2f}',
            monitor=callbacks_config.save_model.monitor,
            save_top_k=1,
            mode=callbacks_config.save_model.mode,
        ),
        EarlyStopping(
            monitor=callbacks_config.early_stop.monitor,
            patience=callbacks_config.early_stop.patience,
            mode=callbacks_config.early_stop.mode,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]


def initialize_trainer(config, callbacks, logger):
    """
    Initialize PyTorch Lightning Trainer.
    """
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        logger=logger,
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
    )


def initialize_data_module(config):
    """
    Initialize the SegmentationDataModule with dataset and augmentation settings.
    """
    class_weight_path = Path(config.directories.logs) / "class_weights.json"
    return SegmentationDataModule(
        dataset_path=config.dataset.dataset_path,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        model_name=config.training.model_name,
        id2label=config.dataset.id2label,
        transform=Augmentation(),
        weighting_strategy=config.loss.weighting_strategy,
        class_weights_path=class_weight_path,
        ignore_index=config.loss.ignore_index,
    )


def initialize_model(config, class_weights):
    """
    Initialize the Segformer model with the provided configuration and class weights.
    """
    return SegformerFinetuner(
        id2label=config.dataset.id2label,
        model_name=config.training.model_name,
        class_weight=class_weights,
        lr=float(config.training.learning_rate),
        alpha=float(config.loss.alpha),
        beta=float(config.loss.beta),
        ignore_index=config.loss.ignore_index,
        weight_decay=float(config.training.weight_decay),
        max_epochs=config.training.max_epochs,
    )


def train(config, resume_version=None):
    """
    Train the Segformer model with the provided configuration.
    """
    # Prepare paths
    paths = prepare_paths(config, resume_version)

    # Initialize components
    logger = initialize_logger(paths["logs_dir"], resume_version)
    data_module = initialize_data_module(config)
    model = initialize_model(config, data_module.class_weights)
    callbacks = initialize_callbacks(
        paths["pretrained_dir"], paths["checkpoint_dir"],
        callbacks_config=config.callbacks
    )
    trainer = initialize_trainer(config, callbacks, logger)

    # Resolve checkpoint path if resuming from a checkpoint
    resume_checkpoint = find_checkpoint(
        config, resume_version) if resume_version else None

    # Log the training configuration
    print(f"Start training:")
    print(f"SegFormer model: {config.training.model_name}")
    print(f"Training parameters:"
          f"Learning rate: {config.training.learning_rate}, "
          f"Batch size: {config.training.batch_size}, "
          f"Weight decay: {config.training.weight_decay}",
          f"Max epochs: {config.training.max_epochs}")
    print("Cross-Entropy - Dice Loss:\n"
          f"Ignore index: {config.loss.ignore_index}, "
          f"Weights: {config.loss.weighting_strategy}, "
          f"Alpha (Cross-Entropy): {config.loss.alpha}, "
          f"Beta (Dice): {config.loss.beta}")

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)
