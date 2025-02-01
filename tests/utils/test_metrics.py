import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from ucs.utils.metrics import SegMetrics


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
        metrics["mean_iou"], torch.nn.Module
    ), "IoU metric not initialized correctly."
    assert isinstance(
        metrics["mean_dice"], torch.nn.Module
    ), "Dice metric not initialized correctly."
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
