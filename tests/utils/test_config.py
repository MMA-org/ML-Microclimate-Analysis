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
    with open(config_file, "w") as f:
        yaml.dump(MOCK_CONFIG, f)
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


def test_directory_creation(mock_config_file, tmp_path, cleanup_directories):
    """
    Test automatic directory creation for paths in the `project` section.
    """
    # Update MOCK_CONFIG paths to absolute paths in tmp_path
    project_dirs = {key: str(tmp_path / key) for key in MOCK_CONFIG["project"]}
    MOCK_CONFIG["project"] = project_dirs

    # Rewrite the config file with updated paths
    with open(mock_config_file, "w") as f:
        yaml.dump(MOCK_CONFIG, f)

    # Register directories for cleanup
    cleanup_directories(project_dirs.values())

    # Create config and check directory creation
    config = Config(config_path=mock_config_file, create_dirs=True)

    for dir_path in project_dirs.values():
        assert Path(dir_path).exists(
        ), f"Directory '{dir_path}' was not created."


@pytest.mark.parametrize("create_dirs", [True, False])
def test_from_dict_with_and_without_creation(tmp_path, create_dirs):
    """
    Test creating a Config object from a dictionary with and without directory creation.
    """
    project_dirs = {
        "models_dir": str(tmp_path / "models"),
        "pretrained_dir": str(tmp_path / "pretrained"),
        "logs_dir": str(tmp_path / "logs"),
        "results_dir": str(tmp_path / "results"),
    }

    # Create config from dictionary
    config = Config.from_dict(
        {"project": project_dirs}, create_dirs=create_dirs)

    assert config.project.models_dir == str(tmp_path / "models")
    assert config.project.results_dir == str(tmp_path / "results")

    for dir_path in project_dirs.values():
        if create_dirs:
            assert Path(dir_path).exists(
            ), f"Directory '{dir_path}' was not created."
        else:
            assert not Path(dir_path).exists(
            ), f"Directory '{dir_path}' should not have been created."


def test_no_directory_creation(mock_config_file, tmp_path, cleanup_directories):
    """
    Test disabling automatic directory creation.
    """
    project_dirs = {key: str(tmp_path / key) for key in MOCK_CONFIG["project"]}

    # Rewrite the config file with updated paths
    with open(mock_config_file, "w") as f:
        yaml.dump({"project": project_dirs}, f)

    # Register directories for cleanup
    cleanup_directories(project_dirs.values())

    # Create config without creating directories
    config = Config(config_path=mock_config_file, create_dirs=False)

    for dir_path in project_dirs.values():
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

    config = Config.from_dict(config_dict)

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
