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


@pytest.fixture
def class_weights():
    """Fixture for sample class weights."""
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def weights_dir(tmp_path):
    """Fixture for creating a temporary directory for weights."""
    return tmp_path / "weights"


def test_save_class_weights(weights_dir, class_weights):
    """Test saving class weights to a file."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    strategy = "raw"

    # Save weights
    save_class_weights(weights_dir, class_weights, strategy)

    # Verify file exists
    weights_file = weights_dir / "class_weights.pt"
    assert weights_file.exists(), "Weights file was not created."

    # Verify file content
    saved_data = torch.load(weights_file, weights_only=True)
    assert torch.equal(
        saved_data["class_weights"], class_weights
    ), "Saved weights do not match the expected values."
    assert (
        saved_data["strategy"] == strategy
    ), "Saved strategy does not match the expected value."


def test_load_class_weights(weights_dir, class_weights):
    """Test loading class weights from a file."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    strategy = "raw"

    # Save weights for testing load
    save_class_weights(weights_dir, class_weights, strategy)

    # Load weights
    loaded_weights = load_class_weights(weights_dir, strategy)

    assert loaded_weights is not None, "Failed to load class weights."
    assert torch.equal(
        loaded_weights, class_weights
    ), "Loaded weights do not match the expected values."


def test_load_with_wrong_strategy(weights_dir, class_weights):
    """Test that loading with the wrong strategy returns None."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    correct_strategy = "balanced"
    wrong_strategy = "random"

    # Save with correct strategy
    save_class_weights(weights_dir, class_weights, correct_strategy)

    # Attempt to load with the wrong strategy
    loaded_weights = load_class_weights(weights_dir, wrong_strategy)

    assert (
        loaded_weights is None
    ), "Loaded weights even though the strategy didn't match."


def test_load_nonexistent_file(tmp_path):
    """Test loading weights from a non-existent file."""
    nonexistent_dir = tmp_path / "nonexistent"
    loaded_weights = load_class_weights(nonexistent_dir, "any_strategy")

    assert loaded_weights is None, "Loaded weights from a non-existent file."
