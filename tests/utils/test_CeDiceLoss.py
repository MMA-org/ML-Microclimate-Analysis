import pytest
import torch
from utils.metrics import CeDiceLoss


@pytest.fixture
def mock_inputs():
    # Mock inputs: Batch of logits with 2 samples, 3 classes, and 4x4 spatial resolution
    return torch.randn((2, 3, 4, 4), requires_grad=True)


@pytest.fixture
def mock_targets():
    # Mock targets: Batch of ground truth labels
    return torch.randint(0, 3, (2, 4, 4))


@pytest.fixture
def mock_weights():
    # Mock class weights
    return [0.2, 0.5, 0.3]


def test_initialization_with_weights(mock_weights):
    """Test initialization with custom weights."""
    loss_fn = CeDiceLoss(num_classes=3, weights=mock_weights)
    assert torch.is_tensor(
        loss_fn.weights), "Weights should be converted to a tensor."


def test_initialization_without_weights():
    """Test initialization without weights."""
    loss_fn = CeDiceLoss(num_classes=3)
    assert loss_fn.weights == None


def test_forward_pass(mock_inputs, mock_targets):
    """Test forward pass to compute the combined loss."""
    loss_fn = CeDiceLoss(num_classes=3)
    loss = loss_fn(mock_inputs, mock_targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."
    assert loss.item() > 0, "Loss value should be positive."


def test_forward_pass_with_weights(mock_inputs, mock_targets, mock_weights):
    """Test forward pass with custom class weights."""
    loss_fn = CeDiceLoss(num_classes=3, weights=mock_weights)
    loss = loss_fn(mock_inputs, mock_targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."
    assert loss.item() > 0, "Loss value should be positive."


def test_ignore_index(mock_inputs, mock_targets):
    """Test behavior when ignore_index is set."""
    ignore_index = 2
    mock_targets[0, :, :] = ignore_index  # Set one sample to ignore_index
    loss_fn = CeDiceLoss(num_classes=3, ignore_index=ignore_index)
    loss = loss_fn(mock_inputs, mock_targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."
    assert loss.item() > 0, "Loss value should be positive."


def test_alpha_beta_scaling(mock_inputs, mock_targets):
    """Test scaling of CrossEntropy and Dice loss with alpha and beta."""
    loss_fn = CeDiceLoss(num_classes=3, alpha=0.3, beta=0.7)
    loss = loss_fn(mock_inputs, mock_targets)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."
    assert loss.item() > 0, "Loss value should be positive."
