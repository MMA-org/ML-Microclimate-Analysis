from dataclasses import field
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic.dataclasses import dataclass


@dataclass
class CallbacksConfig:
    early_stop_patience: int = 5
    early_stop_monitor: str = "val_loss"
    early_stop_mode: str = "min"
    save_model_monitor: str = "val_mean_iou"
    save_model_mode: str = "max"


@dataclass
class TrainingConfig:  # pylint: disable=too-many-instance-attributes
    model_name: str = "b0"
    max_epochs: int = 50
    learning_rate: float = 2e-5
    weight_decay: float = 1e-3
    ignore_index: Optional[int] = 0  # int | None
    # Options: "none", "balanced", "max", "sum", or "raw"
    weighting_strategy: str = "raw"
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
class DatasetConfig:
    dataset_path: str = "erikpinhasov/landcover_dataset"
    batch_size: int = 16
    num_workers: int = 8
    do_reduce_labels: bool = False
    pin_memory: bool = True
    model_name: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    directories: DirectoriesConfig = field(default_factory=DirectoriesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)

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

        config.dataset.model_name = config.training.model_name
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
                    break

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
            print("Created directories:\n" + "\n".join(created_dirs))
