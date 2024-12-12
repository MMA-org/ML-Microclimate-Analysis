import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch
from utils import Config, find_checkpoint,  save_confusion_matrix_plot, apply_color_map, plot_image_and_mask, save_class_weights, load_class_weights
import json
import matplotlib
import torch
matplotlib.use('Agg')


@pytest.fixture
def mock_config(tmp_path):
    """
    Fixture for a mock configuration object.
    """
    config_dict = {
        "project": {
            "logs_dir": str(tmp_path / "logs")
        }
    }
    return Config.from_dict(config_dict)


def test_find_checkpoint(mock_config, tmp_path):
    """
    Test the `find_checkpoint` function.
    """
    version = "0"
    checkpoint_dir = tmp_path / "logs" / f"version_{version}" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    # Create a mock checkpoint file
    checkpoint_file = checkpoint_dir / "mock_checkpoint.ckpt"
    checkpoint_file.touch()

    # Check that the correct checkpoint path is returned
    checkpoint_path = find_checkpoint(mock_config, version)
    assert checkpoint_path == str(
        checkpoint_file), "Checkpoint file path mismatch."

    # Test when the directory doesn't exist
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        find_checkpoint(mock_config, "999")

    # Test when no checkpoint files exist in the directory
    empty_dir = tmp_path / "logs" / "version_empty" / "checkpoints"
    empty_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No checkpoint files found"):
        find_checkpoint(mock_config, "empty")


def test_save_confusion_matrix_plot(tmp_path):
    """Test saving a confusion matrix plot to a file."""
    y_true = [0, 1, 2, 1, 0, 2, 1]
    y_pred = [0, 1, 2, 0, 0, 1, 2]
    labels = ["Class 0", "Class 1", "Class 2"]
    save_path = tmp_path / "confusion_matrix.png"

    save_confusion_matrix_plot(y_true, y_pred, labels, save_path)

    # Check if the file is created
    assert save_path.exists(), "Confusion matrix plot was not saved."


def test_apply_color_map():
    """
    Test the `apply_color_map` function.
    """
    mask = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [1, 0, 3]
    ])

    color_mask = apply_color_map(mask)

    # Assert that the color mask has the correct shape and values
    assert color_mask.shape == (3, 3, 3), "Color mask shape mismatch."
    assert np.array_equal(
        color_mask[0, 0], (255, 255, 255)), "Background color mismatch."
    assert np.array_equal(
        color_mask[0, 1], (255, 0, 0)), "Building color mismatch."
    assert np.array_equal(
        color_mask[0, 2], (255, 255, 0)), "Road color mismatch."


def test_plot_image_and_mask(tmp_path):
    """
    Test the `plot_image_and_mask` function.
    """
    # Create a mock image file
    image_path = tmp_path / "mock_image.png"
    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(mock_image).save(image_path)

    # Create a mock mask
    mask = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [1, 0, 3]
    ])

    # Mock plt.show() to avoid rendering during tests
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_image_and_mask(str(image_path), mask)
        mock_show.assert_called_once()


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
