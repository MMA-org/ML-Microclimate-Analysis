from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model.lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import lc_id2label
from utils.metrics import compute_class_weights
from utils.save_pretrained_callback import SavePretrainedCallback
import torch


def train(config, resume_checkpoint=None):
    """
    Train the Segformer model with the provided configuration.

    Args:
        config: Configuration object containing training parameters.
        resume_checkpoint: Optional checkpoint file to resume training from.
    """

    # Prepare paths
    pretrained_dir = Path(config.project.pretrained_dir)
    logs_dir = Path(config.project.logs_dir)
    version = f"version_{len(list(logs_dir.iterdir()))}"  # fetch version
    checkpoint_dir = logs_dir / version / "checkpoints"
    pretrained_dir = pretrained_dir / version

    # Prepare dataloaders
    loader = Loader(config)
    train_loader = loader.get_dataloader("train", shuffle=True)
    val_loader = loader.get_dataloader("validation")

    # Compute class weights if enabled
    class_weights = (
        compute_class_weights(train_loader, len(lc_id2label))
        if config.training.do_class_weight
        else None
    )

    # Initialize the model
    model = SegformerFinetuner(
        id2label=lc_id2label,
        model_name=config.training.model_name,
        class_weight=class_weights,
    )

    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=config.project.logs_dir, default_hp_metric=False)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.training.patience,
        mode="min",
    )
    pretrained_callback = SavePretrainedCallback(
        pretrained_dir, checkpoint_callback)

    # Initialize Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback,
                   pretrained_callback, early_stop_callback],
        log_every_n_steps=config.training.log_every_n_steps,
    )

    # Resolve resume checkpoint path
    if resume_checkpoint:
        resume_checkpoint_path = checkpoint_dir / resume_checkpoint
        resume_checkpoint = resume_checkpoint_path.resolve(
        ) if not resume_checkpoint_path.is_absolute() else resume_checkpoint_path

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
