import argparse
from model.lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import Config, lc_id2label
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Create necessary directories
    models_dir = Path(config.project.models_dir)
    pretrained_dir = Path(config.project.pretrained_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataloaders
    loader = Loader(config)
    train_loader = loader.get_dataloader("train", shuffle=True)
    val_loader = loader.get_dataloader("validation")

    # Initialize model
    model = SegformerFinetuner(id2label=lc_id2label)

    # Define the logger
    logger = TensorBoardLogger(
        save_dir=Path(config.project.logs_dir),  # Log directory
        default_hp_metric=False
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.save_dir}/{logger.version}/checkpoints",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.training.patience,
        mode="min"
    )

    # Initialize Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the trained model
    model.save_pretrained_model(pretrained_dir, logger.version)


if __name__ == "__main__":
    main()
