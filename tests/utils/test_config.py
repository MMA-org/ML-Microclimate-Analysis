import pytest
import yaml
import tempfile
import os
from utils.config import Config


def test_load_default_config():
    config = Config.load_config()

    assert config.dataset.batch_size == 16
    assert config.training.learning_rate == 2e-5
    assert config.callbacks.early_stop_patience == 5
    assert config.training.weighting_strategy == "raw"


def test_load_yaml_config():
    yaml_data = {
        "dataset": {
            "batch_size": 64
        },
        "training": {
            "learning_rate": 0.0005
        },
        "callbacks": {
            "early_stop_patience": 10
        }
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        temp_file_path = temp_file.name
        with open(temp_file_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file)

    config = Config.load_config(temp_file_path)

    assert config.dataset.batch_size == 64
    assert config.training.learning_rate == 0.0005
    assert config.callbacks.early_stop_patience == 10
    assert config.dataset.num_workers == 8

    os.remove(temp_file_path)


def test_load_with_cli_overrides():
    config = Config.load_config(batch_size=128, early_stop_patience=20)

    assert config.dataset.batch_size == 128
    assert config.callbacks.early_stop_patience == 20
    assert config.training.learning_rate == 2e-5


def test_yaml_and_cli_combined():
    yaml_data = {
        "dataset": {
            "batch_size": 64
        },
        "training": {
            "learning_rate": 0.0005
        },
        "callbacks": {
            "early_stop_patience": 10
        }
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        temp_file_path = temp_file.name
        with open(temp_file_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file)

    config = Config.load_config(
        temp_file_path, batch_size=128, early_stop_patience=20)

    assert config.dataset.batch_size == 128
    assert config.callbacks.early_stop_patience == 20
    assert config.training.learning_rate == 0.0005
    assert config.dataset.num_workers == 8

    os.remove(temp_file_path)


def test_missing_yaml_file():
    config = Config.load_config("non_existent.yaml")

    assert config.dataset.batch_size == 16
    assert config.training.learning_rate == 2e-5


def test_invalid_cli_argument():
    config = Config.load_config(invalid_arg=123)

    assert not hasattr(config, "invalid_arg")
    assert config.dataset.batch_size == 16
