import pytest
from unittest.mock import MagicMock, patch
from utils.save_pretrained_callback import SavePretrainedCallback
from pathlib import Path


@pytest.fixture
def mock_checkpoint_callback():
    """
    Fixture for a mock checkpoint callback.
    """
    callback = MagicMock()
    callback.best_model_path = "mock_best_model.ckpt"
    return callback


@pytest.fixture
def mock_trainer():
    """
    Fixture for a mock trainer object.
    """
    return MagicMock()


@pytest.fixture
def mock_pl_module():
    """
    Fixture for a mock PyTorch Lightning module.
    """
    module = MagicMock()
    module.save_pretrained_model = MagicMock()
    return module


def test_save_pretrained_callback_initialization(mock_checkpoint_callback):
    """
    Test that SavePretrainedCallback initializes correctly.
    """
    callback = SavePretrainedCallback(
        "mock_pretrained_dir", mock_checkpoint_callback)
    assert callback.pretrained_dir == Path("mock_pretrained_dir")
    assert callback.checkpoint_callback == mock_checkpoint_callback
    assert callback.last_best_model_path is None, "Expected last_best_model_path to be None."


@patch("pathlib.Path.exists")
def test_on_validation_epoch_end_saves_model(mock_exists, mock_checkpoint_callback, mock_trainer, mock_pl_module):
    """
    Test that the callback saves the model when a new best checkpoint exists.
    """
    mock_exists.return_value = True  # Simulate that the checkpoint file exists
    callback = SavePretrainedCallback(
        "mock_pretrained_dir", mock_checkpoint_callback)

    # Simulate validation epoch end
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Check that save_pretrained_model was called
    mock_pl_module.save_pretrained_model.assert_called_once_with(
        Path("mock_pretrained_dir"), Path("mock_best_model.ckpt")
    )


def test_on_validation_epoch_end_no_new_checkpoint(mock_checkpoint_callback, mock_trainer, mock_pl_module):
    """
    Test that the callback does nothing if there is no new best checkpoint.
    """
    callback = SavePretrainedCallback(
        "mock_pretrained_dir", mock_checkpoint_callback)
    # Simulate no change in best checkpoint
    callback.last_best_model_path = "mock_best_model.ckpt"

    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Check that save_pretrained_model was not called
    mock_pl_module.save_pretrained_model.assert_not_called()


@patch("pathlib.Path.exists")
def test_on_validation_epoch_end_missing_checkpoint(mock_exists, mock_checkpoint_callback, mock_trainer, mock_pl_module):
    """
    Test that the callback handles missing checkpoint files gracefully.
    """
    mock_exists.return_value = False  # Simulate that the checkpoint file does not exist
    callback = SavePretrainedCallback(
        "mock_pretrained_dir", mock_checkpoint_callback)

    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Check that save_pretrained_model was not called
    mock_pl_module.save_pretrained_model.assert_not_called()
