import pytest
from pathlib import Path
from utils import Config
import yaml
import shutil

MOCK_CONFIG = {
    "project": {
        "models_dir": "models",
        "pretrained_dir": "pretrained_models",
        "logs_dir": "logs",
        "results_dir": "results",
    }
}


@pytest.fixture
def mock_config_file(tmp_path):
    """
    Create a temporary mock configuration file.
    """
    config_file = tmp_path / "config.yaml"

    # Ensure paths in MOCK_CONFIG are absolute paths based on tmp_path
    absolute_paths = {key: str(tmp_path / value)
                      for key, value in MOCK_CONFIG["project"].items()}

    config_dict = {"project": absolute_paths}

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

    # Cleanup registered directories
    for dir_path in created_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)


@pytest.mark.parametrize("create_dirs", [True, False])
def test_load_config(mock_config_file, tmp_path, create_dirs, cleanup_directories):
    """
    Test automatic directory creation for paths in the `project` section.
    """
    # Use paths already updated in mock_config_file
    project_dirs = list(MOCK_CONFIG["project"].values())

    # Register directories for cleanup
    cleanup_directories(project_dirs)

    # Create config and check directory creation
    config = Config(config_path=mock_config_file, create_dirs=create_dirs)

    # Check if the directories are created under the tmp_path
    for dir_path in project_dirs:
        # Join tmp_path with each directory from the config
        # Resolves relative paths correctly
        expected_path = tmp_path / Path(dir_path).name

        if create_dirs:
            assert expected_path.exists(
            ), f"Directory '{expected_path}' was not created."
        else:
            assert not expected_path.exists(
            ), f"Directory '{expected_path}' should not have been created."


@pytest.mark.parametrize("create_dirs", [True, False])
def test_from_dict(tmp_path, create_dirs, cleanup_directories):
    """
    Test creating a Config object from a dictionary with and without directory creation.
    """
    config_dict = MOCK_CONFIG["project"]
    project_dirs = list(config_dict.values())

    # Register directories for cleanup
    cleanup_directories(project_dirs)

    # Create config from dictionary
    config = Config.from_dict(
        {"project": config_dict}, create_dirs=create_dirs)

    for dir_path in project_dirs:
        if create_dirs:
            assert Path(dir_path).exists(
            ), f"Directory '{dir_path}' was not created."
        else:
            assert not Path(dir_path).exists(
            ), f"Directory '{dir_path}' should not have been created."


def test_missing_config_key():
    """
    Test that accessing a non-existent configuration key raises AttributeError.
    """
    config_dict = {
        "project": {
            "logs_dir": "logs",
            "pretrained_dir": "pretrained_models",
        }
    }

    config = Config.from_dict(config_dict, False)

    # Try to access a non-existent key
    with pytest.raises(AttributeError, match="Configuration key 'non_existent_key' not found."):
        _ = config.non_existent_key


def test_get_existing_key():
    """
    Test the `get` method with existing keys.
    """
    config_dict = {
        "project": {
            "logs_dir": "logs",
            "pretrained_dir": "pretrained_models",
        }
    }

    config = Config.from_dict(config_dict, False)

    # Test existing key path
    assert config.get(
        "project", "logs_dir") == "logs", "Failed to retrieve an existing key."


def test_get_missing_key():
    """
    Test the `get` method with a missing key and a default value.
    """
    config_dict = {
        "project": {
            "logs_dir": "logs",
            "pretrained_dir": "pretrained_models",
        }
    }

    config = Config.from_dict(config_dict, False)

    # Test non-existent key path
    default_value = "default_value"
    assert config.get("project", "non_existent_key", default=default_value) == default_value, (
        "Did not return the default value for a missing key."
    )
