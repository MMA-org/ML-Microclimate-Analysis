import matplotlib
import pytest
import torch
from ucs.utils import load_class_weights, save_class_weights
from ucs.utils.config import Config

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
            "checkpoint_dir": str(checkpoint_dir),  # Path for checkpoint_dir
        }
    }
    return Config.from_dict(config_dict)


def test_load_class_weights(tmp_path):
    """Test loading class weights from a file."""
    # Prepare test file
    weights_file = tmp_path / "class_weights.pt"
    class_weights = torch.tensor([1.0, 2.0, 3.0])
    torch.save(class_weights, weights_file)

    # Load weights and assert correctness
    loaded_weights = torch.tensor(load_class_weights(tmp_path))
    assert torch.equal(
        loaded_weights, class_weights
    ), "Loaded weights do not match the expected values."


def test_save_class_weights(tmp_path):
    """Test saving class weights to a file."""
    # Prepare test weights and file
    weights_file = tmp_path / "class_weights.pt"
    class_weights = torch.tensor([1.0, 2.0, 3.0])

    # Save weights
    save_class_weights(tmp_path, class_weights)

    # Verify file content
    saved_weights = torch.load(weights_file, weights_only=True)
    assert torch.equal(
        saved_weights, class_weights
    ), "Saved weights do not match the expected values."
