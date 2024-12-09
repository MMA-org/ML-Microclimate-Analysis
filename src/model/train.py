

def train(config, resume_checkpoint=None):
    from model.lightning_model import SegformerFinetuner
    from data.loader import Loader
    from utils import lc_id2label
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pathlib import Path
    from utils.save_pretrained_callback import SavePretrainedCallback

    # Prepare dataloaders
    loader = Loader(config)
    pretrained_dir = Path(config.project.pretrained_dir)
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

    version = f"version_{logger.version}"
    checkpoint_dir = f"{logger.save_dir}/version/checkpoints"
    pretrained_dir = pretrained_dir/version

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.training.patience,
        mode="min"
    )

    pretrained_callback = SavePretrainedCallback(
        pretrained_dir, checkpoint_callback)

    # Initialize Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=config.training.max_epochs,
        callbacks=[checkpoint_callback,
                   pretrained_callback, early_stop_callback],
        log_every_n_steps=config.training.log_every_n_steps,
    )

    # Train the model

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)

    # Save the trained model

    pretrained_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_model(pretrained_dir, checkpoint_callback)
