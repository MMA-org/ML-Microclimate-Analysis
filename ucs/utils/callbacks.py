from pytorch_lightning.callbacks import ModelCheckpoint, Callback


class SaveModel(ModelCheckpoint):
    """
    A PyTorch Lightning callback to save the pretrained model after checkpointing.

    This callback extends the functionality of ModelCheckpoint to save the pretrained 
    model to a specified directory in addition to saving the training checkpoint.

    Args:
        pretrained_path (str): The directory where the pretrained model will be saved.
        *args: Variable length argument list for the base ModelCheckpoint class.
        **kwargs: Arbitrary keyword arguments for the base ModelCheckpoint class.
    """

    def __init__(self, pretrained_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_path = pretrained_path

    def _save_checkpoint(self, trainer, filepath):
        """
        Overrides the base method to save the pretrained model alongside the checkpoint.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            filepath (str): Path to the checkpoint file being saved.
        """
        super()._save_checkpoint(trainer, filepath)

        if trainer.is_global_zero:  # main process
            trainer.lightning_module.save_pretrained_model(
                self.pretrained_path)


class UnfreezeOnPlateau(Callback):
    """
    A PyTorch Lightning callback to unfreeze the layers of the model 
    when the monitored metric plateaus. Stops monitoring after unfreezing.

    Args:
        monitor (str): The metric to monitor for improvements (e.g., "val_loss").
        mode (str): One of {"min", "max"}. In "min" mode, lower values are better. 
                    In "max" mode, higher values are better.
        patience (int): Number of epochs to wait for improvement before unfreezing layers.
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
    """

    def __init__(self, monitor="val_loss", mode="min", patience=3, delta=0.0):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.best_value = None
        self.epochs_without_improvement = 0
        self.unfreeze_done = False

        # Validate mode
        if self.mode not in {"min", "max"}:
            raise ValueError(
                f"Invalid mode: {self.mode}. Choose 'min' or 'max'.")

    def on_validation_end(self, trainer, pl_module):
        """
        Triggered at the end of validation. Checks if the monitored metric has plateaued
        and unfreezes the specified layers if necessary.

        Args:
            trainer (Trainer): The Lightning trainer instance.
            pl_module (LightningModule): The Lightning module being trained.
        """
        if self.unfreeze_done:
            return  # Stop monitoring after unfreezing

        current_value = self._get_current_metric(trainer)
        if current_value is None:
            return  # Skip if the monitored metric is not logged

        if self._is_improvement(current_value):
            self._update_best_value(current_value)
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience and pl_module.is_layers_freeze:
                self._unfreeze_layers(trainer, pl_module)

    def _get_current_metric(self, trainer):
        """
        Retrieve the current value of the monitored metric.

        Args:
            trainer (Trainer): The Lightning trainer instance.

        Returns:
            float or None: The current value of the monitored metric.
        """
        return trainer.callback_metrics.get(self.monitor)

    def _is_improvement(self, current_value):
        """
        Check if the current value represents a meaningful improvement.

        Args:
            current_value (float): The current value of the monitored metric.

        Returns:
            bool: True if the improvement is significant based on the mode and delta.
        """
        if self.best_value is None:
            return True  # Treat the first value as an improvement

        improvement = current_value - self.best_value
        if self.mode == "min":
            return improvement <= -self.delta
        else:  # mode == "max"
            return improvement >= self.delta

    def _update_best_value(self, current_value):
        """
        Update the best value of the monitored metric.

        Args:
            current_value (float): The current value of the monitored metric.
        """
        self.best_value = current_value
        self.epochs_without_improvement = 0

    def _unfreeze_layers(self, trainer, pl_module):
        """
        Unfreezes the specified layers in the Lightning module and stops further monitoring.

        Args:
            pl_module (LightningModule): The Lightning module being trained.
        """
        pl_module.unfreeze_encoder_layers()
        self.unfreeze_done = True
        self._log_event(trainer,
                        f"Unfrozen encoder layers due to plateau at epoch {trainer.current_epoch}. "
                        f"Patience: {self.patience}, Best {self.monitor}: {self.best_value:.4f}"
                        )

    def _log_event(self, trainer, message):
        """
        Log an event message to the console and experiment logger.

        Args:
            trainer (Trainer): The Lightning trainer instance.
            message (str): The message to log.
        """
        if trainer.is_global_zero:
            print(message)  # Immediate visibility
            if trainer.logger:
                logger_method = getattr(
                    trainer.logger.experiment, "log_text", None)
                if callable(logger_method):
                    try:
                        logger_method(message, step=trainer.current_epoch)
                    except Exception:
                        pass  # Skip if logging fails
