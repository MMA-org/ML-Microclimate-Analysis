import yaml
from pathlib import Path

from typing import Any, Dict


import copy


class Config:
    """
    ConfigLoader class to load configurations from a default YAML file,
    a custom YAML file, or argparse arguments.
    """

    def __init__(self, config_path="config.yaml"):
        """
        """
        self.config = default_config
        self.__merge_yaml__(config_path)

    def __load_yaml__(self, yaml_file: str) -> dict:
        """Helper function to load a YAML file."""
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

    def __merge_yaml__(self, yaml_path: str):
        """
        Merge custom YAML configurations with the default configurations.
        :param custom_yaml: Path to the custom YAML file.
        """
        if not Path(yaml_path).exists():
            print("Config is not provided use default configurations.")
            return self.config
        custom_config = self.__load_yaml__(yaml_path)
        self.config = self.__merge_dicts__(self.config, custom_config)

    @staticmethod
    def __merge_dicts__(base: dict, override: dict) -> dict:
        """
        Recursively merge two dictionaries.
        :param base: The base dictionary.
        :param override: The dictionary with override values.
        :return: A merged dictionary.
        """
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = Config.__merge_dicts__(merged[key], value)
            else:
                merged[key] = value
        return merged

    def __getattr__(self, name):
        if name in self.config:
            value = self.config.get(name)
            if value == "None":
                return None
            if name == "id2label":
                return value
            if isinstance(value, dict):
                return Config.from_dict(value, False)
            return value
        raise AttributeError(f"Configuration key '{name}' not found.")

    @staticmethod
    def from_dict(config_dict, create_dirs=True):
        """Create a Config object from a dictionary."""
        config = Config.__new__(
            Config)  # Create a new instance without calling __init__
        config.config = config_dict
        if create_dirs:
            config.__create_directories__()
        return config

    def load_from_args(self, args_dict: dict):
        """
        Update the configuration with command-line arguments.
        """
        for arg_name, val in args_dict.items():
            if val is not None and arg_name in arg_to_key_map:
                keys = arg_to_key_map[arg_name]
                sub_config = self.config
                for key in keys[:-1]:
                    sub_config = sub_config.setdefault(key, {})
                sub_config[keys[-1]] = val

    def get(self, *keys, default=None):
        """
        Get nested configuration values with a fallback default.
        """
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default

    def to_dict(self):
        """
        Get the entire configuration as a dictionary.
        """
        return self.config

    def __create_directories__(self):
        """
        Automatically create directories specified in the `project` section and print only newly created directories.
        """
        project_config = self.config.get("project", {})
        created_dirs = [
            dir_path for dir_path in project_config.values()
            if not Path(dir_path).exists() and Path(dir_path).mkdir(parents=True, exist_ok=True) is None
        ]
        if created_dirs:
            print(f"Created directories: {', '.join(created_dirs)}")


default_config = {
    "dataset": {
        "dataset_path": "erikpinhasov/landcover_dataset"
    },
    "project": {
        "models_dir": "models",
        "pretrained_dir": "models/pretrained_models",
        "logs_dir": "models/logs",
        "results_dir": "results",
    },
    "training": {
        "batch_size": 32,
        "max_epochs": 50,
        "log_every_n_steps": 10,
        "learning_rate": 2e-5,
        "model_name": "b1",
        "early_stop": {
            "patience": 10,
        },
        "focal_loss": {
            "gamma": 2,
            "alpha": None,
            "ignore_index": 0,
            "weights": {
                "class_weights": True,
                "normalize": "balanced",
            },
        }
    }
}
arg_to_key_map = {
    # Dataset
    "dataset_path": ["dataset", "dataset_path"],

    # Project
    "models_dir": ["project", "models_dir"],
    "pretrained_dir": ["project", "pretrained_dir"],
    "logs_dir": ["project", "logs_dir"],
    "results_dir": ["project", "results_dir"],

    # Training
    "batch_size": ["training", "batch_size"],
    "max_epochs": ["training", "max_epochs"],
    "log_step": ["training", "log_every_n_steps"],
    "lr": ["training", "learning_rate"],
    "model_name": ["training", "model_name"],

    # Early stopping
    "stop_patience": ["training", "early_stop", "patience"],

    # Focal loss
    "gamma": ["training", "focal_loss", "gamma"],
    "alpha": ["training", "focal_loss", "alpha"],
    "ignore_index": ["training", "focal_loss", "ignore_index"],

    # Focal loss weights
    "class_weights": ["training", "focal_loss", "weights", "class_weights"],
    "normalize": ["training", "focal_loss", "weights", "normalize"]
}
