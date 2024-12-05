
import argparse
from lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import load_config, lc_id2label
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Create models dir
    models_dir = Path(config["project"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataloaders
    loader = Loader(config)
    train_loader = loader.get_dataloader("train", shuffle=True)
    val_loader = loader.get_dataloader("validation")

    # Initialize model
    model = SegformerFinetuner(id2label=lc_id2label)

    # Set up Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["project"]["checkpoint_dir"], monitor="val_loss", save_top_k=1, mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    trainer = Trainer(
        default_root_dir=config["project"]["logs_dir"],
        max_epochs=config["training"]["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10)

    trainer.fit(model, train_loader, val_loader)
    model.save_pretrained_model(config["project"]["model_dir"])


if __name__ == "__main__":
    main()
