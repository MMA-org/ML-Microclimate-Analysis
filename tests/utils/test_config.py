import pytest
from pathlib import Path
from utils.config import Config
import yaml
import shutil

MOCK_CONFIG = {
    "dataset": {
        "id2label": {
            0: "example1",
            1: "example2"
        }
    },
    "directories": {
        "models": "models",
        "pretrained": "pretrained_models",
        "logs": "logs",
        "checkpoints"
        "results": "results",
    }
}


@pytest.fixture
def mock_config_file(tmp_path):
    """
    Create a temporary mock configuration file with absolute paths.
    """
    config_file = tmp_path / "config.yaml"

    # Update paths to be absolute
    absolute_paths = {key: str(tmp_path / value)
                      for key, value in MOCK_CONFIG["directories"].items()}

    config_dict = {
        "directories": absolute_paths,
        "dataset": MOCK_CONFIG["dataset"]
    }

    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    return config_file


@pytest.fixture
def cleanup_directories():
    """
    Helper fixture to clean up directories after tests.
    """
    created_dirs = []

    def register(dirs):
        created_dirs.extend(dirs)

    yield register

    for dir_path in created_dirs:
        shutil.rmtree(dir_path, ignore_errors=True)


@pytest.mark.parametrize("create_dirs", [True, False])
def test_from_dict(tmp_path, create_dirs, cleanup_directories):
    """
    Test Config creation from a dictionary.
    """
    # Update directories directories to be relative to tmp_path
    directories_dirs = {key: str(tmp_path / value)
                        for key, value in MOCK_CONFIG["directories"].items()}

    # Register directories for cleanup
    cleanup_directories(list(directories_dirs.values()))

    # Create Config from dictionary
    config = Config.from_dict(
        {"directories": directories_dirs}, create_dirs=create_dirs)

    for dir_path in directories_dirs.values():
        dir_path_obj = Path(dir_path)
        if create_dirs:
            assert dir_path_obj.exists(
            ), f"Directory '{dir_path_obj}' was not created."
        else:
            assert not dir_path_obj.exists(
            ), f"Directory '{dir_path_obj}' should not have been created."


def test_missing_config_key():
    """
    Test accessing a non-existent configuration key.
    """
    config = Config.from_dict(
        {"directories": {"logs": "logs"}}, create_dirs=False)

    with pytest.raises(AttributeError, match="Configuration key 'non_existent_key' not found."):
        _ = config.non_existent_key


def test_get_existing_key():
    """
    Test retrieving an existing key with `get`.
    """
    config = Config.from_dict(
        {"directories": {"logs": "logs"}}, create_dirs=False)
    assert config.get(
        "directories", "logs") == "logs", "Failed to retrieve an existing key."


def test_get_missing_key():
    """
    Test retrieving a missing key with a default value using `get`.
    """
    config = Config.from_dict(
        {"directories": {"logs": "logs"}}, create_dirs=False)
    default_value = "default_value"
    assert config.get("directories", "non_existent_key", default=default_value) == default_value, (
        "Did not return the default value for a missing key."
    )
