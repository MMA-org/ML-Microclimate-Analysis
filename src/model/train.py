from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model.lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import load_class_weights, save_class_weights, get_next_version, find_checkpoint
from utils.metrics import compute_class_weights
from utils.save_pretrained_callback import SavePretrainedCallback
from data.transform import Augmentation
import torch


def prepare_paths(config, resume_version=None):
    """Prepare directory paths for logging and checkpoints."""
    # Get the base directories
    pretrained_dir = Path(config.directories.pretrained)
    logs_dir = Path(config.directories.logs)

    # Determine the version for logging/checkpoints
    version = f"version_{resume_version}" if resume_version else get_next_version(
        logs_dir)

    # Define paths for checkpoints and pretrained models
    checkpoint_dir = Path(config.directories.checkpoints) / version
    pretrained_dir = pretrained_dir / version

    return pretrained_dir, logs_dir, checkpoint_dir


def prepare_dataloaders(config):
    """Prepare train and validation dataloaders."""
    loader = Loader(config)
    train_loader = loader.get_dataloader(
        "train", shuffle=True, transform=Augmentation())
    val_loader = loader.get_dataloader("validation")

    return loader, train_loader, val_loader


def prepare_class_weights(config, loader):
    """
    Compute or load precomputed class weights based on the configuration.

    Args:
        config (Config): Configuration object containing paths and settings.
        train_loader (DataLoader): DataLoader for the training dataset.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    # Check if class weighting is enabled
    class_weights = config.loss.weights
    num_classes = len(config.dataset.id2label)

    if not class_weights:
        return None  # Return None if class weighting is disabled

    # Define the path for class weights file
    weights_file = Path(config.directories.logs) / "class_weights.json"
    normalize = config.loss.normalize
    ignore_index = config.loss.ignore_index

    # Load or compute class weights
    if weights_file.exists():
        class_weights = load_class_weights(weights_file)
    else:
        train_loader = loader.get_dataloader("train")
        class_weights = compute_class_weights(
            train_loader, num_classes, normalize=normalize, ignore_index=ignore_index)
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
    # Set the accelerator and number of devices
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


def resolve_checkpoint_path(config, resume_version: int) -> Path:
    """
    Resolve the path for resuming training from a checkpoint.

    Args:
        config: Configuration object.
        resume_version (int): The version folder name (e.g., "version_0").

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
        config: (Config) Configuration object containing training parameters.
        resume_version (int): Optional checkpoint file to resume training from.
    """
    # Prepare paths for logging, checkpoints, and pretrained models
    pretrained_dir, logs_dir, checkpoint_dir = prepare_paths(
        config, resume_version)

    # Initialize logger for TensorBoard
    logger = TensorBoardLogger(
        save_dir=logs_dir,
        version=f"version_{resume_version}" if resume_version else None,
        default_hp_metric=False
    )

    # Prepare data loaders for training and validation
    loader, train_loader, val_loader = prepare_dataloaders(config)

    # Prepare class weights for training
    class_weights = prepare_class_weights(config, loader)

    # Initialize the Segformer model
    model = SegformerFinetuner(
        id2label=config.dataset.id2label,
        model_name=config.training.model_name,
        class_weight=class_weights,  # alpha/class weights
        lr=float(config.training.learning_rate),
        alpha=float(config.loss.alpha),
        beta=float(config.loss.beta),
        ignore_index=config.loss.ignore_index
    )

    # Initialize callbacks (Checkpoint, EarlyStopping, SavePretrained)
    callbacks = initialize_callbacks(
        pretrained_dir, checkpoint_dir, config.training.early_stop.patience)

    # Initialize trainer
    trainer = initialize_trainer(config, callbacks, logger)

    # Resolve checkpoint path if resuming from a checkpoint
    resume_checkpoint = resolve_checkpoint_path(config, resume_version)

    print(f"Start training:")
    print(f"SegFormer model: {config.training.model_name}, "
          f"Learning rate: {config.training.learning_rate}, "
          f"Batch size: {config.training.batch_size}, ")
    print("Croos-Entropy - Dice Loss:\n"
          f"Ignore index: {config.loss.ignore_index}, "
          f"Weights: {config.loss.weights}, "
          f"Normalize method: {config.loss.normalize}, "
          f"Alpha (Cross-Entropy): {config.loss.alpha}, "
          f"Beta (Dice): {config.loss.beta}")

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
