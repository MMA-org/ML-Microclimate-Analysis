from unittest.mock import MagicMock, patch

import pytest

from ucs.utils.callbacks import SaveModel


@pytest.fixture
def mock_trainer():
    """
    Fixture for a mock trainer object.
    """
    trainer = MagicMock()
    trainer.is_global_zero = True  # Simulate the main process
    return trainer


@pytest.fixture
def mock_pl_module():
    """
    Fixture for a mock PyTorch Lightning module.
    """
    module = MagicMock()
    module.save_pretrained_model = MagicMock()
    return module


def test_save_model_initialization():
    """
    Test that SaveModel initializes correctly.
    """
    callback = SaveModel(pretrained_dir="mock_pretrained_dir")
    assert callback.pretrained_dir == "mock_pretrained_dir"


@patch("pytorch_lightning.callbacks.ModelCheckpoint._save_checkpoint")
def test_save_model_calls_super_save_checkpoint(
    mock_super_save, mock_trainer, mock_pl_module
):
    """
    Test that SaveModel calls the parent _save_checkpoint method.
    """
    callback = SaveModel(pretrained_dir="mock_pretrained_dir")
    callback._save_checkpoint(mock_trainer, "mock_checkpoint.ckpt")

    # Check that the parent method was called
    mock_super_save.assert_called_once_with(mock_trainer, "mock_checkpoint.ckpt")


@patch("pytorch_lightning.callbacks.ModelCheckpoint._save_checkpoint")
def test_save_model_saves_pretrained_on_global_zero(
    mock_super_save, mock_trainer, mock_pl_module
):
    """
    Test that SaveModel saves the pretrained model only on the main process (is_global_zero).
    """
    mock_trainer.lightning_module = mock_pl_module  # Attach the mock module
    callback = SaveModel(pretrained_dir="mock_pretrained_dir")
    callback._save_checkpoint(mock_trainer, "mock_checkpoint.ckpt")

    # Ensure save_pretrained_model was called
    mock_pl_module.save_pretrained_model.assert_called_once_with("mock_pretrained_dir")


@patch("pytorch_lightning.callbacks.ModelCheckpoint._save_checkpoint")
def test_save_model_does_not_save_pretrained_on_non_global_zero(
    mock_super_save, mock_trainer, mock_pl_module
):
    """
    Test that SaveModel does not save the pretrained model if not global zero.
    """
    mock_trainer.is_global_zero = False  # Simulate a non-main process
    mock_trainer.lightning_module = mock_pl_module  # Attach the mock module
    callback = SaveModel(pretrained_dir="mock_pretrained_dir")
    callback._save_checkpoint(mock_trainer, "mock_checkpoint.ckpt")

    # Ensure save_pretrained_model was not called
    mock_pl_module.save_pretrained_model.assert_not_called()
