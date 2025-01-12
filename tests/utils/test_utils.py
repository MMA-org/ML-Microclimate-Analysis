import pytest
import numpy as np
from utils import Config, find_checkpoint, save_class_weights, load_class_weights
import json
import matplotlib
from pathlib import Path
import torch
matplotlib.use('Agg')


@pytest.fixture
def mock_config(tmp_path):
    """
    Fixture for a mock configuration object.
    """
    # Define the directories
    logs_dir = tmp_path / "logs"
    checkpoint_dir = logs_dir / "checkpoints" / "version_0"

    # Create the directories
    checkpoint_dir.mkdir(parents=True)

    # Create the mock config dictionary
    config_dict = {
        "directories": {
            "logs": str(logs_dir),  # Path for logs_dir
            "checkpoint_dir": str(checkpoint_dir)  # Path for checkpoint_dir
        }
    }
    return Config.from_dict(config_dict)


def test_find_checkpoint(mock_config, tmp_path):
    """
    Test the `find_checkpoint` function.
    """
    version = "0"

    # Create the checkpoint directory as expected by the function
    cpkt_dir = Path(mock_config.directories.checkpoint_dir)
    cpkt_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Create the checkpoint file
    checkpoint_file = cpkt_dir / "mock_checkpoint.ckpt"
    checkpoint_file.touch()  # This creates the file

    # Test: Check that the correct checkpoint path is returned
    checkpoint_path = find_checkpoint(mock_config, version)
    assert checkpoint_path == checkpoint_file.resolve(
    ), f"Expected {checkpoint_file.resolve()}, got {checkpoint_path}"

    # Test when the directory doesn't exist
    with pytest.raises(SystemExit) as excinfo:
        find_checkpoint(mock_config, 999)  # Non-existing version
    # Check that exit code is 3 for CheckpointDirectoryError
    assert excinfo.value.code == 3

    # Test when no checkpoint files exist in the directory
    empty_dir = tmp_path / "logs" / "checkpoints" / "version_empty"
    empty_dir.mkdir(parents=True)
    with pytest.raises(SystemExit) as excinfo:
        # Version with no checkpoints
        find_checkpoint(mock_config, "empty")
    # Check that exit code is 2 for CheckpointNotFoundError
    assert excinfo.value.code == 2

    # Test when multiple checkpoint files exist in the directory
    multiple_dir = tmp_path / "logs" / "checkpoints" / "version_multiple"
    multiple_dir.mkdir(parents=True)
    (multiple_dir / "checkpoint1.ckpt").touch()
    (multiple_dir / "checkpoint2.ckpt").touch()
    with pytest.raises(SystemExit) as excinfo:
        # Version with multiple checkpoints
        find_checkpoint(mock_config, "multiple")
    # Check that exit code is 4 for MultipleCheckpointsError
    assert excinfo.value.code == 4


def test_load_class_weights(tmp_path):
    """Test loading class weights from a file."""
    # Prepare test file
    weights_file = tmp_path / "class_weights.json"
    class_weights = torch.tensor([1.0, 2.0, 3.0])
    with weights_file.open("w") as f:
        json.dump(class_weights.tolist(), f)

    # Load weights and assert correctness
    loaded_weights = torch.tensor(load_class_weights(weights_file))
    assert torch.equal(
        loaded_weights, class_weights), "Loaded weights do not match the expected values."


def test_save_class_weights(tmp_path):
    """Test saving class weights to a file."""
    # Prepare test weights and file
    weights_file = tmp_path / "class_weights.json"
    class_weights = torch.tensor([1.0, 2.0, 3.0])

    # Save weights
    save_class_weights(weights_file, class_weights)

    # Verify file content
    with weights_file.open("r") as f:
        saved_weights = torch.tensor(json.load(f))
    assert torch.equal(
        saved_weights, class_weights), "Saved weights do not match the expected values."
