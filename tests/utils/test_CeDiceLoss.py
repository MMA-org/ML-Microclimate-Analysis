import pytest
import torch
import torch.nn as nn
from utils.metrics import CeDiceLoss


@pytest.fixture
def sample_data():
    # Create sample inputs and targets
    # Batch size 4, 3 classes, 256x256 resolution
    inputs = torch.randn(4, 3, 256, 256)
    targets = torch.randint(0, 3, (4, 256, 256))  # Ground truth labels
    return inputs, targets


@pytest.fixture
def loss_function():
    # Create a CeDiceLoss instance
    return CeDiceLoss(num_classes=3, alpha=0.5, weights=[1.0, 2.0, 1.5])


def test_loss_computation(loss_function, sample_data):
    """Test that the combined loss is computed without errors."""
    inputs, targets = sample_data
    loss = loss_function(inputs, targets)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
    assert loss.item() > 0, "Loss value should be positive."


def test_ignore_index_behavior():
    """Test that the ignore_index argument is handled correctly."""
    loss_function = CeDiceLoss(num_classes=3, ignore_index=1)

    # Create inputs and targets with ignored labels
    inputs = torch.randn(4, 3, 256, 256)
    targets = torch.randint(0, 3, (4, 256, 256))
    targets[0, :, :] = 1  # Set the first batch to the ignored class

    loss = loss_function(inputs, targets)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
    assert loss.item() >= 0, "Loss value should be non-negative."


def test_weights_application():
    """Test that class weights are applied correctly."""
    weights = [0.5, 2.0, 1.0]
    loss_function = CeDiceLoss(num_classes=3, weights=weights)

    inputs = torch.randn(4, 3, 256, 256)
    targets = torch.randint(0, 3, (4, 256, 256))

    loss = loss_function(inputs, targets)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
    assert loss.item() > 0, "Loss value should be positive."

    # Ensure weights are reflected in CrossEntropyLoss
    assert torch.equal(loss_function.ce_loss.weight, torch.tensor(
        weights, dtype=torch.float)), "Weights should match."


def test_alpha_scaling():
    """Test the impact of alpha scaling on the loss."""
    inputs = torch.randn(4, 3, 256, 256)
    targets = torch.randint(0, 3, (4, 256, 256))

    loss_function1 = CeDiceLoss(
        num_classes=3, alpha=1.0)  # Only Cross-Entropy
    loss_function2 = CeDiceLoss(
        num_classes=3, alpha=0.0)  # Only Dice

    ce_loss = loss_function1(inputs, targets)
    dice_loss = loss_function2(inputs, targets)

    assert ce_loss.item() > 0, "Cross-Entropy loss should be positive."
    assert dice_loss.item() > 0, "Dice loss should be positive."


def test_no_weights():
    """Test the loss function when no weights are provided."""
    loss_function = CeDiceLoss(num_classes=3)

    inputs = torch.randn(4, 3, 256, 256)
    targets = torch.randint(0, 3, (4, 256, 256))

    loss = loss_function(inputs, targets)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
    assert loss.item() > 0, "Loss value should be positive."
