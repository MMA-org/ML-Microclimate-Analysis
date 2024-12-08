import argparse
from model.lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import Config, lc_id2label
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path


def parse_config():
    """
    Parse command-line arguments and load the configuration file.
    """
    parser = argparse.ArgumentParser(description="Train SegFormer")
    parser.add_argument("--config", type=str,
                        default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, nargs='?', const=None,
                        help="Path to checkpoint to resume training (optional)")

    args = parser.parse_args()
    return Config(args.config), args.resume


def main():
    # Load configuration and optional checkpoint path
    config, resume_checkpoint = parse_config()

    # Prepare dataloaders
    loader = Loader(config)
    train_loader = loader.get_dataloader("train", shuffle=True)
    val_loader = loader.get_dataloader("validation")

    # Initialize model
    model = SegformerFinetuner(
        id2label=lc_id2label,
        model_name=config.training.model_name,
    )

    # Define the logger
    logger = TensorBoardLogger(
        save_dir=config.project.logs_dir,
        default_hp_metric=False
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.save_dir}/version_{logger.version}/checkpoints",
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
        log_every_n_steps=config.training.log_every_n_steps,
    )

    # Train the model

    trainer.fit(model, train_loader, val_loader)

    # Save the trained model
    pretrained_dir = Path(config.project.pretrained_dir) / \
        f"version_{logger.version}"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_model(pretrained_dir)


if __name__ == "__main__":
    main()
