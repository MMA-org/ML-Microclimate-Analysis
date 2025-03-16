# util/metric.py
from typing import Literal

import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.segmentation import DiceScore, MeanIoU
from tqdm import tqdm

from ucs.core.errors import NormalizeError
from ucs.utils.functional import one_hot


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, eps=1e-6, weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        if weights is not None:
            weights = weights / weights.sum()  # Normalize
            weights = weights.view(1, -1)
            self.register_buffer("weights", weights)
        else:
            self.weights = None  # Handle None case

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss, ignoring specified class index.

        Args:
            logits (torch.Tensor): Probability tensor of shape (B, C, H, W).
            targets (torch.Tensor): Target tensor of shape (B, H, W) with class indices.

        Returns:
            torch.Tensor: Dice loss.
        """
        num_classes = logits.shape[1]  # Get number of valid classes
        probs = torch.softmax(logits, dim=1).float()
        # Convert predictions & targets to one-hot encoding
        targets_one_hot = one_hot(targets, num_classes, self.ignore_index)
        # Compute intersection and union correctly
        intersection = (probs * targets_one_hot).sum(dim=(2, 3)).to(probs.dtype)
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        # Avoid division by zero when union is empty
        dice_score = (2.0 * intersection) / (union + self.eps)
        if self.weights is not None:
            dice_score = dice_score * self.weights
        # Compute mean Dice Loss
        return 1 - dice_score.mean()


class CeDiceLoss(nn.Module):
    def __init__(self, alpha=0.7, weights=None, ignore_index=255, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.ignore_index = ignore_index
        self.weights = self._initialize_weights(weights)

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            weight=self.weights,
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index, weights=self.weights)

    def forward(self, inputs, targets):
        """
        Compute combined CE and Dice loss.
        - CE Loss uses raw logits (as required).
        - Dice Loss requires softmax probabilities.

        Args:
            inputs (torch.Tensor): Logits of shape (B, C, H, W)
            targets (torch.Tensor): Ground truth labels of shape (B, H, W)

        Returns:
            torch.Tensor: Weighted sum of CE and Dice Loss.
        """
        ce_loss = self.ce_loss(inputs, targets)  # Use raw logits for CE Loss
        dice_loss = self.dice_loss(inputs, targets)

        return self.alpha * ce_loss + self.beta * dice_loss

    def _initialize_weights(self, weights):
        """
        Validate and initialize class weights.

        Args:
            weights (list, np.ndarray, torch.Tensor, optional): Class weights.
            num_classes (int): Number of classes.

        Returns:
            torch.Tensor: Validated weight tensor or None.
        """
        if weights is None:
            return None  # No weights provided, return None

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)

        return weights


class SegMetrics(MetricCollection):
    """
    A utility class for handling segmentation task metrics.
    Computes IoU (Jaccard Index) and Dice coefficient.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        prefix (str): Prefix for metric names.
        ignore_index (int, optional): Index to ignore in the target labels. Default is `255`.
        average (str, optional): Averaging method. Default is `"macro"`.

    Attributes:
        num_classes (int): The number of classes.
        ignore_index (int): The index to ignore in the target.
        prefix (str): Prefix used for logging metrics.
    """

    def __init__(self, num_classes, prefix, ignore_index=255, average="macro"):
        self.num_classes = num_classes
        self.prefix = prefix
        self.ignore_index = ignore_index
        self.average = average

        metrics = {
            "mean_iou": MeanIoU(num_classes=num_classes),
            "mean_dice": DiceScore(num_classes=num_classes, average=average),
        }

        super().__init__(metrics, prefix=f"{prefix}/")

    def forward(self, preds, targets):
        preds = one_hot(preds, self.num_classes, ignore_index=self.ignore_index)
        targets = one_hot(targets, self.num_classes, ignore_index=self.ignore_index)
        return super().forward(preds, targets)

    def update(self, preds, targets):
        preds = one_hot(preds, self.num_classes, ignore_index=self.ignore_index)
        targets = one_hot(targets, self.num_classes, ignore_index=self.ignore_index)
        return super().update(preds, targets)


class TestMetrics(SegMetrics):
    """
    A utility class for evaluating segmentation models.
    Extends `SegMetrics` by adding confusion matrix, accuracy, precision, and recall.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        prefix (str): Prefix for metric names.
        ignore_index (int, optional): Index to ignore in the target labels. Default is `None`.
        average (str, optional): Averaging method. Default is `"weighted"`.

    Attributes:
        num_classes (int): The number of classes.
        ignore_index (int): The index to ignore in the target.
        prefix (str): Prefix used for logging metrics.
    """

    def __init__(self, num_classes, prefix, ignore_index=255, average="weighted"):
        # Base metrics from SegMetrics
        super().__init__(
            num_classes, prefix, ignore_index=ignore_index, average=average
        )

        # Additional test-specific metrics
        test_metrics = {
            "accuracy": MulticlassAccuracy(
                num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
            "precision": MulticlassPrecision(
                num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
            "recall": MulticlassRecall(
                num_classes=num_classes, average=average, ignore_index=ignore_index
            ),
        }

        # Merge metrics before initialization
        all_metrics = {**test_metrics}

        # Add additional metrics
        self.add_metrics(all_metrics)


def compute_class_weights(
    dataloader,
    num_classes,
    mask_key="labels",
    normalize: Literal["sum", "max", "raw", "balanced"] = "raw",
    ignore_index=255,
):
    """
    Compute class weights based on the frequencies of each class in the dataset.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the dataset.
        num_classes (int): Total number of classes.
        mask_key (str): Key to access masks/labels in the dataloader's batch.
        normalize (str): Method to normalize weights:
            - "sum": Scales weights to sum to 1.
            - "max": Scales weights so the maximum weight is 1.
            - "raw": Leaves weights unnormalized.
            - "balanced": Ensures equal contribution from all classes.
        ignore_index (int, optional): Class index to ignore in weight computation.

    Returns:
        torch.Tensor: Computed class weights of shape (num_classes,).

    Raises:
        NormalizeError: If the normalization method is unsupported.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.int64)

    for batch in tqdm(dataloader, desc="Compute class weights"):
        masks = batch[mask_key]
        masks = masks.view(-1)  # Flatten masks into 1D

        # Extract valid pixels; already 1D
        valid_labels = masks[masks != ignore_index]

        counts = torch.bincount(valid_labels.int(), minlength=num_classes)
        class_counts += counts

    total_pixels = class_counts.sum()
    class_weights = torch.where(
        class_counts > 0,
        1 / torch.log(1.02 + (class_counts / total_pixels) + 1e-6),
        torch.tensor(0.0, dtype=torch.float32),
    )

    # Compute class weights based on the normalization strategy
    if normalize == "sum":
        class_weights /= class_weights.sum()

    elif normalize == "max":
        class_weights /= class_weights.max()

    elif normalize == "balanced":
        class_weights = class_weights * (num_classes / class_weights.sum())

    elif normalize == "raw":
        pass
    else:
        raise NormalizeError(normalize)

    return class_weights.to(torch.float32)
