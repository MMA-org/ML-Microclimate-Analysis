from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


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
            trainer.lightning_module.save_pretrained_model(
                self.pretrained_dir)


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
            if self.epochs_without_improvement >= self.patience:
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
        message = f"UnfreezeOnPlateau: Layers unfrozen at epoch {trainer.current_epoch}. Patience: {self.patience}, Best {self.monitor}: {self.best_value:.4f}"

        self._log_event(trainer, message)

    def _log_event(self, trainer, message):
        """
        Log an event message to the console and experiment logger.

        Args:
            trainer (Trainer): The Lightning trainer instance.
            message (str): The message to log.
        """
        if trainer.is_global_zero and trainer.logger:
            print(f"\n{message}\n")
            if hasattr(trainer.logger.experiment, "log"):
                trainer.logger.experiment.log(
                    {"event_message": message, "epoch": trainer.current_epoch})
            # Handle TensorBoardLogger
            elif hasattr(trainer.logger.experiment, "add_text"):
                trainer.logger.experiment.add_text(
                    "event_logs", message, global_step=trainer.current_epoch)
            # Handle MLFlowLogger
            elif hasattr(trainer.logger.experiment, "set_tags"):
                trainer.logger.experiment.set_tags({"event_message": message})
            # Fallback
            else:
                trainer.logger.log_metrics(
                    {"event_message": message}, step=trainer.current_epoch)


class LossAdjustmentCallback(Callback):
    """
    A callback to adjust the loss function weights dynamically during training.

    The callback changes the weights of the loss components (Cross-Entropy and Dice loss)
    over the course of the training process. It starts with pure Cross-Entropy loss during 
    the warmup phase, then transitions smoothly to a balanced combination of Cross-Entropy 
    and Dice loss, and finally focuses on Dice loss towards the end of training.

    The transition between loss components happens in three phases:
    1. **Warmup Phase** (First `warmup_epochs` epochs): Uses only Cross-Entropy loss.
    2. **Transition Phase** (From `warmup_epochs` to `transition_end`): Linearly adjusts from CE loss to a mix of CE and Dice loss.
    3. **Dice Loss Phase** (After `transition_end` epochs): Uses only Dice loss.

    Args:
        warmup_epochs (int): The number of epochs for the warmup phase where only Cross-Entropy loss is used.
                              Default is 10.
        transition_end (int): The epoch at which the transition to pure Dice loss should end. Default is 40.
    """

    def __init__(self, max_epochs=50):
        super().__init__()
        self.warmup_epochs = max_epochs * 0.2
        self.transition_end = max_epochs * 0.8

    def on_train_epoch_start(self, trainer, pl_module):
        """
        This method is called at the start of each epoch to adjust the loss weights
        based on the current epoch number.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The LightningModule for which the callback is applied.

        The loss weights (`alpha` and `beta`) are dynamically set based on the epoch:
        - During the warmup phase (epochs 0 to `warmup_epochs`), only Cross-Entropy loss is used (`alpha=1.0`, `beta=0.0`).
        - In the transition phase (epochs `warmup_epochs` to `transition_end`), the weights linearly shift from Cross-Entropy to a combination of Cross-Entropy and Dice loss.
        - After `transition_end` epochs, only Dice loss is used (`alpha=0.0`, `beta=1.0`).
        """
        current_epoch = trainer.current_epoch

        # Pure CE loss during warmup
        if current_epoch < self.warmup_epochs:
            alpha, beta = 1.0, 0.0
        # Pure Dice loss after transition
        elif self.warmup_epochs <= current_epoch <= self.transition_end:
            progress = progress = (
                current_epoch - self.warmup_epochs) / (self.transition_end - self.warmup_epochs)
            alpha = 1 - progress
            beta = progress
        # Linear transition phase
        else:
            alpha = 0.0
            beta = 1.0

        # Update loss weights
        pl_module.criterion.set_stage(alpha, beta)

        # Log weights
        pl_module.log("alpha", alpha, on_epoch=True, prog_bar=True)
        pl_module.log("beta", beta, on_epoch=True, prog_bar=True)
