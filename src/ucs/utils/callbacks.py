from pytorch_lightning.callbacks import BaseFinetuning, ModelCheckpoint


class SaveModel(ModelCheckpoint):
    """
    A PyTorch Lightning callback to save the pretrained model after checkpointing.

    This callback extends the functionality of ModelCheckpoint to save the pretrained
    model to a specified directory in addition to saving the training checkpoint.

    Args:
        pretrained_dir (str): The directory where the pretrained model will be saved.
        *args: Variable length argument list for the base ModelCheckpoint class.
        **kwargs: Arbitrary keyword arguments for the base ModelCheckpoint class.
    """

    def __init__(self, pretrained_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_dir = pretrained_dir

    def _save_checkpoint(self, trainer, filepath):
        """
        Overrides the base method to save the pretrained model alongside the checkpoint.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            filepath (str): Path to the checkpoint file being saved.
        """
        super()._save_checkpoint(trainer, filepath)

        if trainer.is_global_zero:  # main process
            trainer.lightning_module.save_pretrained_model(self.pretrained_dir)


class GradualUnfreeze(BaseFinetuning):
    def __init__(self, phase1_epoch=2, phase2_epoch=5):
        super().__init__()
        self.phase1_epoch = phase1_epoch
        self.phase2_epoch = phase2_epoch

    def freeze_before_training(self, pl_module):
        """Freeze Patch Embeddings + Blocks [0,1] before training starts."""
        self.freeze(
            pl_module.model.segformer.encoder.patch_embeddings
        )  # Freeze Patch Embeddings
        self.freeze(pl_module.model.segformer.encoder.block[:2])  # Freeze Blocks [0,1]
        self._log(
            pl_module.trainer,
            "Starting training: Patch embeddings & Blocks [0,1] frozen.",
        )

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """Gradually unfreeze the encoder in two phases."""
        if current_epoch == self.phase1_epoch:
            # Unfreeze Block [1], Keep Patch Embeddings + Block [0] Frozen
            self.unfreeze_and_add_param_group(
                pl_module.model.segformer.encoder.block[1],
                optimizer,
                lr=pl_module.hparams.lr_encoder,
            )
            self._log(
                pl_module.trainer,
                f"Phase 1 (Epoch {current_epoch}): Unfreezing Block [1].",
            )

        elif current_epoch == self.phase2_epoch:
            # Unfreeze All Encoder Layers
            self.unfreeze_and_add_param_group(
                pl_module.model.segformer.encoder.block[0],
                optimizer,
                lr=pl_module.hparams.lr_encoder,
            )
            self.unfreeze_and_add_param_group(
                pl_module.model.segformer.encoder.patch_embeddings,
                optimizer,
                lr=pl_module.hparams.lr_encoder,
            )
            self._log(
                pl_module.trainer,
                f"Phase 2 (Epoch {current_epoch}): Unfreezing All Encoder Layers.",
            )

    def _log(self, trainer, message):
        """Logs messages to console and TensorBoard if available."""
        if trainer.is_global_zero:
            if trainer.logger and hasattr(trainer.logger.experiment, "add_text"):
                trainer.logger.experiment.add_text(
                    "GradualFreezing", message, global_step=trainer.global_step
                )
