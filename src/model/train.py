from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model.lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import lc_id2label, load_class_weights, save_class_weights, get_next_version, find_checkpoint
from utils.metrics import compute_class_weights
from utils.save_pretrained_callback import SavePretrainedCallback
import torch


def prepare_paths(config, resume_version=None):
    """Prepare directory paths for logging and checkpoints."""
    pretrained_dir = Path(config.project.pretrained_dir)
    logs_dir = Path(config.project.logs_dir)
    version = f"version_{resume_version}" if resume_version else get_next_version(
        logs_dir)
    checkpoint_dir = logs_dir / "checkpoints" / version
    pretrained_dir = pretrained_dir / version
    return pretrained_dir, logs_dir, checkpoint_dir


def prepare_dataloaders(config):
    """Prepare train and validation dataloaders."""
    loader = Loader(config)
    train_loader = loader.get_dataloader("train", shuffle=True)
    val_loader = loader.get_dataloader("val")
    return train_loader, val_loader


def prepare_class_weights(config, train_loader):
    """
    Compute or load precomputed class weights based on the configuration.

    Args:
        config: Configuration object containing paths and settings.
        train_loader: DataLoader for the training dataset.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    weights_file = Path(config.project.logs_dir) / "class_weights.json"

    normalize_weights = config.training.focal_loss.normalize_weights
    if not config.training.focal_loss.do_class_weight:
        return None  # Return None if class weighting is disabled

    if weights_file.exists():
        class_weights = load_class_weights(weights_file)
    else:
        class_weights = compute_class_weights(
            train_loader, len(lc_id2label), normalize=normalize_weights)
        save_class_weights(weights_file, class_weights)

    return class_weights


def initialize_callbacks(pretrained_dir, checkpoint_dir, patience):
    """Initialize callbacks for model training."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}-{val_mean_iou:.2f}',
        monitor="val_mean_iou",
        save_top_k=1,
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
    )
    pretrained_callback = SavePretrainedCallback(
        pretrained_dir, checkpoint_callback)
    return [checkpoint_callback, early_stop_callback, pretrained_callback]


def initialize_trainer(config, callbacks, logger):
    """Initialize the PyTorch Lightning Trainer."""
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    return Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=config.training.log_every_n_steps,
    )


def resolve_checkpoint_path(config, resume_version: str) -> Path:
    """
    Resolve the path for resuming training from a checkpoint.

    Args:
        config: Configuration object.
        resume_version (str): The version folder name (e.g., "version_0").

    Returns:
        Path: Absolute path to the checkpoint file or None if no resume version is provided.
    """
    if resume_version:
        return find_checkpoint(config, resume_version)
    return None


def train(config, resume_version=None):
    """
    Train the Segformer model with the provided configuration.

    Args:
        config: Configuration object containing training parameters.
        resume_checkpoint: Optional checkpoint file to resume training from.
    """
    # Prepare paths
    pretrained_dir, logs_dir, checkpoint_dir = prepare_paths(
        config, resume_version)

    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=logs_dir,
        version=f"version_{resume_version}" if resume_version else None,
        default_hp_metric=False
    )

    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(config)

    # Compute class weights

    class_weights = prepare_class_weights(config, train_loader)

    # Initialize the model
    model = SegformerFinetuner(
        id2label=lc_id2label,
        model_name=config.training.model_name,
        class_weight=class_weights,
        lr=float(config.training.learning_rate),
        gamma=float(config.training.focal_loss.gamma),
        ignore_index=config.training.focal_loss.ignore_index
    )

    # Initialize callbacks
    callbacks = initialize_callbacks(
        pretrained_dir, checkpoint_dir, config.training.early_stop.patience)

    # Initialize trainer
    trainer = initialize_trainer(config, callbacks, logger)

    # Resolve checkpoint path
    resume_checkpoint = resolve_checkpoint_path(
        config, resume_version)

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
