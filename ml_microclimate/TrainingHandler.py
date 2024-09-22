# filename: training_handler.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os

class TrainingHandler:
    def __init__(self, model, train_dataloader, val_dataloader, max_epochs=50, checkpoint_dir="../model/checkpoints/"):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        # Ensure the checkpoint directory exists
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set float precision for matmul
        torch.set_float32_matmul_precision('medium')

        # Set up callbacks
        self.early_stop_callback = EarlyStopping(
            monitor="val_loss",  # Monitor IoU for early stopping
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="min",  # Stop when IoU stops improving
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}",  # Save based on IoU
            save_top_k=1,  # Save the best model based on IoU
            monitor="val_loss",  # Monitor IoU for checkpointing
            mode="min",  # Save model with the highest IoU
        )

        # Determine accelerator (GPU or CPU)
        self.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    def get_trainer(self):
        # Set up the PyTorch Lightning trainer
        trainer = pl.Trainer(
            accelerator=self.accelerator,  # Use GPU if available, otherwise CPU
            devices=1 if self.accelerator == 'gpu' else None,  # Use one GPU or default to CPU
            callbacks=[self.early_stop_callback,
                       self.checkpoint_callback],  # All callbacks
            max_epochs=self.max_epochs,  # Maximum number of epochs to train
            # Perform validation after each epoch
            val_check_interval=len(self.train_dataloader),
            log_every_n_steps=10,  # Log more frequently to monitor training progress
        )
        return trainer

    def train(self):
        # Get the trainer and start training
        trainer = self.get_trainer()
        trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
