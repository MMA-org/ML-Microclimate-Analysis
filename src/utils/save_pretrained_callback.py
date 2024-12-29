from pytorch_lightning.callbacks import Callback
from pathlib import Path


class SavePretrainedCallback(Callback):
    """
    Callback to save a Hugging Face pretrained model when a new best checkpoint is created.
    """

    def __init__(self, pretrained_dir, checkpoint_callback):
        """
        Args:
            pretrained_dir (str): Directory to save the Hugging Face model.
            checkpoint_callback (ModelCheckpoint): ModelCheckpoint instance to track the best checkpoint.
        """
        super().__init__()
        self.pretrained_dir = Path(pretrained_dir)
        self.checkpoint_callback = checkpoint_callback
        self.last_best_model_path = None

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch.
        """
        current_best_model_path = self.checkpoint_callback.best_model_path

        # If a new best checkpoint exists, save the pretrained model
        if current_best_model_path and current_best_model_path != self.last_best_model_path:
            checkpoint_path = Path(current_best_model_path)

            # Ensure the checkpoint file exists
            if not checkpoint_path.exists():
                print(f"Checkpoint file not found: {current_best_model_path}")
                return

            # Update the last best model path
            self.last_best_model_path = current_best_model_path

            # Save the Hugging Face model
            pl_module.save_pretrained_model(
                self.pretrained_dir, checkpoint_path)
