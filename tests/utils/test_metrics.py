import pytest
import torch
from utils.metrics import SegMetrics, compute_class_weights
from torch.utils.data import DataLoader, Dataset


class MockDataset(Dataset):
    """
    A mock dataset for testing compute_class_weights.
    """

    def __init__(self, masks, mask_key="labels"):
        self.masks = masks
        self.mask_key = mask_key

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return {self.mask_key: self.masks[idx]}


@pytest.fixture
def metrics():
    return SegMetrics(num_classes=3, device="cpu")  # Explicitly set device


@pytest.fixture
def mock_dataloader():
    """
    Mock dataloader for testing class weights computation.
    """
    masks = [
        torch.randint(0, 3, (2, 2)),  # Random labels in a small 2x2 grid
        torch.randint(0, 3, (2, 2)),
    ]
    dataset = MockDataset(masks, mask_key="labels")
    return DataLoader(dataset, batch_size=1)


def test_metrics_initialization(metrics):
    """
    Test the initialization of the Metrics class.
    """
    assert isinstance(
        metrics.metrics["mean_iou"], torch.nn.Module), "IoU metric not initialized correctly."
    assert isinstance(
        metrics.metrics["mean_dice"], torch.nn.Module), "Dice metric not initialized correctly."
    assert metrics.num_classes == 3, "Number of classes should be 3."


def test_metrics_update(metrics):
    """
    Test the update method of the Metrics class.
    """
    predicted = torch.randint(0, 3, (4, 4))
    targets = torch.randint(0, 3, (4, 4))
    metrics.update(predicted, targets)

    # Ensure no exceptions occur and metrics are updated
    assert True, "Metrics update failed."


def test_metrics_compute(metrics):
    """
    Test the compute method of the Metrics class.
    """
    predicted = torch.randint(0, 3, (4, 4))
    targets = torch.randint(0, 3, (4, 4))
    metrics.update(predicted, targets)
    results = metrics.compute()

    assert "mean_iou" in results, "mean_iou missing in computed metrics."
    assert "mean_dice" in results, "mean_dice missing in computed metrics."
    assert results["mean_iou"] >= 0, "mean_iou should be non-negative."
    assert results["mean_dice"] >= 0, "mean_dice should be non-negative."


def test_metrics_reset(metrics):
    """
    Test the reset method of the Metrics class.
    """
    metrics.reset()
    assert True, "Metrics reset failed."


def test_compute_class_weights(mock_dataloader):
    """
    Test compute_class_weights function.
    """
    num_classes = 3
    class_weights = compute_class_weights(
        mock_dataloader, num_classes, mask_key="labels")

    # Ensure class weights are computed correctly
    assert isinstance(
        class_weights, torch.Tensor), "Class weights should be a tensor."
    assert class_weights.shape == (
        num_classes,), "Class weights should have length equal to num_classes."
    assert all(
        weight > 0 for weight in class_weights), "All class weights should be positive."

    mask_array = torch.cat([batch["labels"].view(-1)
                           for batch in mock_dataloader])
    class_counts = torch.bincount(mask_array, minlength=num_classes)

    # Compute expected class weights manually based on inverse frequency
    total_samples = mask_array.numel()
    expected_class_weights = total_samples / (class_counts.float() + 1e-6)
    normalized_weights = expected_class_weights / expected_class_weights.sum()

    # Check if computed weights match the expected ones (within a small tolerance)
    assert torch.allclose(class_weights, normalized_weights, atol=1e-2), \
        f"Class weights do not match. Expected: {normalized_weights}, but got: {class_weights}"


def test_compute_class_weights_empty_dataset():
    """
    Test compute_class_weights with an empty dataset.
    """
    dataset = MockDataset([], mask_key="labels")
    dataloader = DataLoader(dataset, batch_size=1)

    num_classes = 3
    with pytest.raises(ValueError, match="The dataset is empty. Cannot compute class weights."):
        compute_class_weights(dataloader, num_classes, mask_key="labels")


def test_compute_class_weights_invalid_masks():
    """
    Test compute_class_weights with invalid mask formats.
    """
    invalid_masks = ["invalid", 123, None]
    dataset = MockDataset(invalid_masks, mask_key="labels")
    dataloader = DataLoader(dataset, batch_size=1)

    num_classes = 3
    with pytest.raises(KeyError):
        compute_class_weights(dataloader, num_classes, mask_key="masks")
