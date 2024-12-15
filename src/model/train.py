from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model.lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import lc_id2label, load_class_weights, save_class_weights
from utils.metrics import compute_class_weights
from utils.save_pretrained_callback import SavePretrainedCallback
import torch


def prepare_paths(config):
    """Prepare directory paths for logging and checkpoints."""
    pretrained_dir = Path(config.project.pretrained_dir)
    logs_dir = Path(config.project.logs_dir)
    version = f"version_{len(list(logs_dir.iterdir()))}"  # fetch version
    checkpoint_dir = logs_dir / "checkpoints" / version
    pretrained_dir = pretrained_dir / version
    return pretrained_dir, logs_dir, checkpoint_dir


def prepare_dataloaders(config):
    """Prepare train and validation dataloaders."""
    loader = Loader(config)
    train_loader = loader.get_dataloader("train", shuffle=True)
    val_loader = loader.get_dataloader("val")
    return train_loader, val_loader


def prepare_class_weights(logs_dir: Path, do_class_weight, train_loader):
    """
    Compute or load precomputed class weights based on the configuration.

    Args:
        config: Configuration object containing paths and settings.
        train_loader: DataLoader for the training dataset.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    weights_file = logs_dir / "class_weights.json"

    if not do_class_weight:
        return None  # Return None if class weighting is disabled

    if weights_file.exists():
        class_weights = load_class_weights(weights_file)
    else:
        class_weights = compute_class_weights(train_loader, len(lc_id2label))
        save_class_weights(weights_file, class_weights)

    return class_weights


def initialize_callbacks(pretrained_dir, checkpoint_dir, patience):
    """Initialize callbacks for model training."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
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


def resolve_checkpoint_path(checkpoint_dir, resume_checkpoint):
    """Resolve the path for resuming training from a checkpoint."""
    if resume_checkpoint:
        checkpoint_path = checkpoint_dir / resume_checkpoint
        return checkpoint_path.resolve() if not checkpoint_path.is_absolute() else checkpoint_path
    return None


def train(config, do_class_weight, resume_checkpoint=None):
    """
    Train the Segformer model with the provided configuration.

    Args:
        config: Configuration object containing training parameters.
        resume_checkpoint: Optional checkpoint file to resume training from.
    """
    # Prepare paths
    pretrained_dir, logs_dir, checkpoint_dir = prepare_paths(config)

    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(config)

    # Compute class weights

    class_weights = prepare_class_weights(
        logs_dir, do_class_weight, train_loader)

    # Initialize the model
    model = SegformerFinetuner(
        id2label=lc_id2label,
        model_name=config.training.model_name,
        class_weight=class_weights,
        lr=7.0e-5,
    )

    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=logs_dir, default_hp_metric=False
    )

    # Initialize callbacks
    callbacks = initialize_callbacks(
        pretrained_dir, checkpoint_dir, config.training.patience)

    # Initialize trainer
    trainer = initialize_trainer(config, callbacks, logger)

    # Resolve checkpoint path
    resume_checkpoint = resolve_checkpoint_path(
        checkpoint_dir, resume_checkpoint)

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
