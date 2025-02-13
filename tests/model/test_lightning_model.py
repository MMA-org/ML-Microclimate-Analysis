from unittest.mock import MagicMock

import pytest
import torch

from ucs.model.lightning_model import (
    SegformerFinetuner,
    SegformerForSemanticSegmentation,
)
from ucs.utils.metrics import SegMetrics, TestMetrics


@pytest.fixture
def mock_id2label():
    return {0: "background", 1: "building", 2: "road"}


@pytest.fixture
def mock_model(mock_id2label):
    """
    Create a mock SegformerFinetuner instance.
    """
    model = SegformerFinetuner(id2label=mock_id2label, model_name="b0")
    model.metrics = SegMetrics(num_classes=len(mock_id2label))
    model.log = MagicMock()
    return model


@pytest.fixture
def mock_batch():
    return {
        "pixel_values": torch.rand((2, 3, 512, 512)),  # Batch of 2 images
        "labels": torch.randint(0, 3, (2, 512, 512)),  # Ground truth masks
    }


def test_initialization(mock_model):
    assert mock_model.num_classes == 3
    assert isinstance(mock_model.model, SegformerForSemanticSegmentation)


def test_training_step(mock_model, mock_batch):
    loss = mock_model.training_step(mock_batch, 0)
    assert loss.item() > 0, "Training step loss should be positive."


def test_validation_step(mock_model, mock_batch):
    loss = mock_model.validation_step(mock_batch, 0)
    assert loss.item() > 0, "Validation step loss should be positive."


def test_test_step(mock_model, mock_batch):
    mock_model.on_test_start()
    result = mock_model.test_step(mock_batch, 0)

    # Validate the loss
    assert isinstance(
        result, torch.Tensor
    ), "Test step result should be a torch.Tensor."
    assert result.item() > 0, "Test step loss should be positive."

    # Validate predictions and ground truths are being updated
    assert (
        mock_model.test_results["predictions"].numel() > 0
    ), "Test results should contain predictions."
    assert (
        mock_model.test_results["ground_truths"].numel() > 0
    ), "Test results should contain ground truths."


def test_on_test_start(mock_model):
    mock_model.on_test_start()
    assert isinstance(
        mock_model.metrics, TestMetrics
    ), "Metrics should be an instance of TestMetrics."


def test_on_test_epoch_end(mock_model):
    mock_model.metrics.reset = MagicMock()
    mock_model.on_test_epoch_end()
    mock_model.metrics.reset.assert_called_once()


def test_on_train_epoch_end(mock_model):
    mock_model.metrics.reset = MagicMock()
    mock_model.on_train_epoch_end()
    mock_model.metrics.reset.assert_called_once()


def test_on_validation_epoch_end(mock_model):
    mock_model.metrics.reset = MagicMock()
    mock_model.on_validation_epoch_end()
    mock_model.metrics.reset.assert_called_once()


def test_forward_pass(mock_model, mock_batch):
    images, masks = mock_batch['pixel_values'], mock_batch['labels']
    _, predictions = mock_model(images, masks)
    assert predictions.shape == (
        2,
        512,
        512,
    ), "Output logits shape should match expected shape."


def test_save_pretrained_model(mock_model, tmp_path):
    pretrained_path = tmp_path / "pretrained_model"
    mock_model.save_pretrained_model(pretrained_path)
    assert pretrained_path.exists(), "Pretrained model path should exist."
