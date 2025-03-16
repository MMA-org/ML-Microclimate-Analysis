from dataclasses import asdict, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic.dataclasses import dataclass


@dataclass
class HparamsConfig:
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 1e-3
    backbone: str = "b0"


@dataclass
class CallbacksConfig:
    early_stop_patience: int = 5
    early_stop_monitor: str = "val_loss"
    early_stop_mode: str = "min"
    save_model_monitor: str = "val_mean_iou"
    save_model_mode: str = "max"


@dataclass
class TrainingConfig(HparamsConfig):
    max_epochs: int = 50
    weighting_strategy: str = (
        "raw"  # Options: "none", "balanced", "max", "sum", or "raw"
    )
    alpha: float = 0.7
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            0: "background",
            1: "bareland",
            2: "rangeland",
            3: "developed space",
            4: "road",
            5: "tree",
            6: "water",
            7: "agriculture land",
            8: "buildings",
        }
    )


@dataclass
class DirectoriesConfig:
    models: str = "models"
    pretrained: str = "models/pretrained_models"
    logs: str = "models/logs"
    checkpoints: str = "models/logs/checkpoints"
    results: str = "results"


@dataclass
class DatasetConfig(HparamsConfig):
    dataset_path: str = "erikpinhasov/landcover_dataset"
    num_workers: int = 8
    pin_memory: bool = True
    batch_size: int = field(default=16)
    include_background: bool = True


@dataclass
class LoggersConfig:
    project_name: str = "landcover-seg"
    tags: List[str] = field(default_factory=lambda: ["segmentation", "landcover"])
    notes: str = "Fine-Tune SegFormer with landcover dataset"
    wandb: bool = True
    wandb_watch: bool = True
    wandb_log: str = "all"
    wandb_log_freq: int = 500
    wandb_log_graph: bool = False


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    directories: DirectoriesConfig = field(default_factory=DirectoriesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    loggers: LoggersConfig = field(default_factory=LoggersConfig)
    hparams: HparamsConfig = field(default_factory=HparamsConfig, repr=False)

    @classmethod
    def load_config(cls, config_path: Optional[str] = None, **overrides) -> "Config":
        """
        Load YAML configuration file and apply overrides.

        Args:
            config_path (Optional[str]): Path to the configuration YAML file.
            **overrides: Dictionary of command-line overrides.

        Returns:
            Config: An instance of the Config class with loaded values.
        """
        config = cls()

        if config_path and Path(config_path).resolve().exists():
            with open(config_path, "r", encoding="utf8") as file:
                yaml_data = yaml.safe_load(file) or {}
            config = cls(**yaml_data)

        config._apply_overrides(overrides)
        config._create_directories()
        return config

    def _apply_overrides(self, overrides):
        """
        Applies CLI override values to the configuration.

        Args:
            overrides (dict): Dictionary of values to override.
        """

        for key, value in overrides.items():
            if value is None:
                continue

            for section_name in self.__annotations__:  # Iterate over attributes
                section = getattr(self, section_name)

                if hasattr(
                    section, key
                ):  # Check if the override key exists in the section
                    setattr(section, key, value)

        self.dataset = replace(self.dataset, **asdict(self.hparams))
        self.training = replace(self.training, **asdict(self.hparams))

    def _create_directories(self):
        """
        Creates required directories if they do not exist.

        Prints the directories that were created.
        """

        created_dirs = [
            dir_path
            for dir_path in vars(self.directories).values()
            if not Path(dir_path).exists()
            and Path(dir_path).mkdir(parents=True, exist_ok=True) is None
        ]
        if created_dirs:
            print("Created directories:\n" + " ".join(created_dirs))
