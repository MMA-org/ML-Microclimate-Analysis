import yaml
from pathlib import Path
from utils.errors import ConfigId2LabelError


class Config:
    """
    Encapsulates configuration data, enabling nested attribute-based access.
    Automatically creates directories specified in the `project` section.
    """

    def __init__(self, config_path="config.yaml", create_dirs=True):
        # Load defaults
        self._config = default_config or {}

        # Load user-defined YAML, merging with defaults
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
            self._merge_dicts(self._config, user_config)

        self._ensure_id2label()
        if create_dirs:
            self._create_directories()

    def __getattr__(self, name):
        value = self._config.get(name)
        if value == "None":
            return None
        if isinstance(value, dict):
            return Config.from_dict(value)
        elif value is not None:
            return value
        raise AttributeError(f"Configuration key '{name}' not found.")

    @staticmethod
    def from_dict(config_dict, create_dirs=True):
        """Create a Config object from a dictionary."""
        config = Config.__new__(
            Config)  # Create a new instance without calling __init__
        config._config = config_dict
        if create_dirs:
            config._create_directories()
        return config

    def get(self, *keys, default=None):
        """
        Get nested configuration values with a fallback default.
        """
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default

    def _create_directories(self):
        """
        Automatically create directories specified in the `project` section.
        """
        project_config = self._config.get("project", {})
        created_dirs = []  # Collect created directory paths

        for key, dir_path in project_config.items():
            dir_path = Path(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))  # Add to list

        if created_dirs:
            print(f"Directories created: {' '.join(created_dirs)}")

    def _merge_dicts(self, base, overrides):
        """
        Recursively merge two dictionaries.
        """
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value

    def _ensure_id2label(self):
        """
        Ensures that 'id2label' is present in the 'dataset' section of the config.
        Raises an error if 'id2label' is missing.
        """
        if 'id2label' not in self._config.get('dataset', {}):
            raise ConfigId2LabelError(
                "'id2label' is missing in the configuration file.")


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
            "ignore_index": None,
            "weights": {
                "do_class_weight": True,
                "normalize_weights": True,
            },
        }
    }
}
