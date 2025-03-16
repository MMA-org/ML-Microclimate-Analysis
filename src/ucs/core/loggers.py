from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class Loggers:
    def __init__(self, config, logs_dir, version):
        """
        Initializes TensorBoard and WandB loggers based on configuration.

        Args:
            config (dict): Configuration containing logger settings.
            logs_dir (str): Directory to save logs.
            version (str or int): Version for resuming runs.
        """
        self.config = config
        self.logs_dir = logs_dir
        self.version = version
        self.loggers = {}

        # TensorBoard settings
        self._init_tensorboard_logger()

        # WandB settings (if enabled)
        if self.config.wandb:
            self._init_wandb_logger()

    def _init_tensorboard_logger(self):
        """Initializes the TensorBoard Logger."""
        tb_logger = TensorBoardLogger(
            name=self.config.project_name,
            save_dir=self.logs_dir,
            version=self.version,
            default_hp_metric=False,
        )
        self.loggers["tb"] = tb_logger

    def _init_wandb_logger(self):
        """Initializes the WandB Logger."""

        wandb_logger = WandbLogger(
            project=self.config.project_name,
            name=f"run_{self.version}",
            save_dir=Path(self.logs_dir) / self.config.project_name,
            version=str(self.version),
            id=str(self.version),
            log_model=True,
            resume="allow",
            tags=self.config.tags,
            notes=self.config.notes,
            anonymous=False,
        )
        self.loggers["wandb"] = wandb_logger

    def get_loggers(self):
        """
        Returns all initialized loggers as a dictionary.

        Returns:
            dict: Dictionary of initialized loggers.
        """
        return self.loggers

    def get_logger(self, logger_name):
        """
        Returns a specific logger by name.

        Args:
            logger_name (str): Name of the logger ('tb' or 'wandb').

        Returns:
            Logger object if exists, else raises KeyError.
        """
        if logger_name not in self.loggers:
            raise KeyError(
                f"Logger '{logger_name}' not found. Available loggers: {list(self.loggers.keys())}"
            )
        return self.loggers[logger_name]

    def get_loggers_list(self):
        """
        Returns a list of all initialized loggers.

        Returns:
            list: List of initialized logger instances.
        """
        return list(self.loggers.values())

    def watch(self, model):
        """Watches the model for gradients and parameters using WandB."""
        if "wandb" in self.loggers:
            self.loggers["wandb"].watch(
                model,
                log=self.config.wandb_log,
                log_freq=self.config.wandb_log_freq,
                log_graph=self.config.wandb_log_graph,
            )

    def log_training_details(self, config):
        """
        Logs the training details to TensorBoard.
        """
        # Prepare the log data
        log_text = f"""
        🚀 Start Training - Version_{self.version}
        📂 Dataset: {config.dataset.dataset_path}  

        🔹 Training Parameters:  
            • Model: {config.training.backbone}  
            • Learning Rate: {config.training.learning_rate} 
            • Weight Decay: {config.training.weight_decay}  
            • Batch Size: {config.dataset.batch_size}  
            • Include Background: {config.dataset.include_background}  
            • Max Epochs: {config.training.max_epochs}

        🎯 Cross-Entropy + Dice Loss:  
            • Weighting Strategy: {config.training.weighting_strategy}  
            • alpha: {config.training.alpha}  
            • beta: {1 - config.training.alpha:.1f}
        """.strip()
        # Log to TensorBoard
        if "tb" in self.loggers:
            self.loggers["tb"].experiment.add_text("Training Info", log_text)

        # Also print to console for immediate feedback
        print(log_text)
