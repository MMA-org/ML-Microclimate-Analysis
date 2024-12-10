import pytest
import torch
from unittest.mock import MagicMock
from model.lightning_model import SegformerFinetuner
from utils.metrics import Metrics


@pytest.fixture
def mock_id2label():
    return {0: "background", 1: "building", 2: "road"}


@pytest.fixture
def mock_model(mock_id2label):
    """
    Create a mock SegformerFinetuner instance.
    """
    model = SegformerFinetuner(id2label=mock_id2label, model_name="b0")
    model.metrics = Metrics(num_classes=len(mock_id2label))
    model.log = MagicMock()
    return model


@ pytest.fixture
def mock_batch():
    return {
        "pixel_values": torch.rand((2, 3, 512, 512)),  # Batch of 2 images
        "labels": torch.randint(0, 3, (2, 512, 512)),  # Ground truth masks
    }


def test_initialization(mock_model):
    """
    Test the initialization of the model.
    """
    assert isinstance(mock_model, SegformerFinetuner)
    assert isinstance(mock_model.metrics, Metrics)
    assert mock_model.num_classes == 3, "Number of classes should match `id2label`."
    assert mock_model.training, "Model need to be in train mode on initialization."


def test_forward_pass(mock_model, mock_batch):
    """
    Test the forward pass.
    """
    loss, predictions = mock_model(
        mock_batch["pixel_values"], mock_batch["labels"])
    assert loss > 0, "Loss should be a positive scalar."
    assert predictions.shape == mock_batch["labels"].shape, "Predictions should match label dimensions."


def test_on_train_start(mock_model):
    """
    Test that `on_train_start` sets the model to training mode.
    """
    mock_model.on_train_start()
    assert mock_model.training, "on_train_start did not set model mode to training."


def test_step_logic(mock_model, mock_batch):
    """
    Test the logic for a single training/validation/test step.
    """
    loss = mock_model.training_step(mock_batch, 0)
    assert loss.item() > 0, "Training step loss should be positive."

    loss = mock_model.validation_step(mock_batch, 0)
    assert loss.item() > 0, "Validation step loss should be positive."

    loss = mock_model.test_step(mock_batch, 0)
    assert loss.item() > 0, "Test step loss should be positive."


@ pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_on_epoch_end(mock_model, stage):
    """
    Test metric logging at the end of an epoch.
    """
    # Mock metrics behavior for this test
    mock_model.metrics = MagicMock()
    mock_model.metrics.compute.return_value = {
        "mean_iou": 0.8, "mean_dice": 0.85}
    mock_model.metrics.reset = MagicMock()

    mock_model.on_epoch_end(stage)
    mock_model.metrics.compute.assert_called_once()
    mock_model.log.assert_any_call(
        f"{stage}_mean_iou", 0.8, prog_bar=True, on_epoch=True)
    mock_model.log.assert_any_call(
        f"{stage}_mean_dice", 0.85, prog_bar=True, on_epoch=True)
    mock_model.metrics.reset.assert_called_once()


def test_optimizer_configuration(mock_model):
    """
    Test optimizer configuration.
    """
    optimizer = mock_model.configure_optimizers()
    assert isinstance(
        optimizer, torch.optim.AdamW), "Optimizer should be AdamW."


def test_reset_test_results(mock_model):
    """
    Test resetting test results.
    """
    mock_model.test_results["predictions"] = [1, 2, 3]
    mock_model.reset_test_results()
    assert mock_model.test_results == {
        "predictions": [], "ground_truths": []}, "Test results not reset."


def test_get_test_results(mock_model):
    """
    Test retrieving test results.
    """
    mock_model.test_results = {"predictions": [1, 2], "ground_truths": [3, 4]}
    results = mock_model.get_test_results()
    assert results == {"predictions": [1, 2], "ground_truths": [
        3, 4]}, "Test results mismatch."
