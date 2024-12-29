# util/metric.py
from tqdm import tqdm
from torchmetrics import MetricCollection, JaccardIndex, Dice, Accuracy, Precision, Recall
import torch
import torch.nn as nn
from typing import Optional
import numpy as np


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, num_classes, gamma=2.0, alpha=None, ignore_index=None, reduction='mean'):
        self.ignore_index = ignore_index if ignore_index is not None else -100
        super().__init__(ignore_index=self.ignore_index, reduction='none')
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.init_alpha(alpha)

    def forward(self, input_, target):
        self.alpha = self.alpha.to(input_.device)
        cross_entropy = super().forward(input_, target)
        pt = torch.exp(-cross_entropy)
        at = self.alpha[target]

        loss = at * ((1-pt)**self.gamma) * cross_entropy

        valid_mask = (target != self.ignore_index)
        valid_loss = loss[valid_mask]

        if self.reduction == 'mean':
            return valid_loss.mean()
        elif self.reduction == 'sum':
            return valid_loss.sum()
        else:
            return loss

    def init_alpha(self, alpha):
        if alpha is None:
            self.alpha = torch.ones(self.num_classes, dtype=torch.float)
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(
                self.num_classes, dtype=torch.float) * alpha
        if isinstance(alpha, (np.ndarray, list)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        if isinstance(alpha, torch.Tensor):
            self.alpha = alpha


class SegMetrics(MetricCollection):
    """
    A utility class to handle metrics for segmentation tasks.
    Provides functionality for IoU (Jaccard Index) and Dice coefficient calculation.
    """

    def __init__(self, num_classes, device="cpu", ignore_index: Optional[int] = None):
        """
        Initialize metrics.

        Args:
            num_classes (int): Number of classes in the segmentation task.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = {
            "mean_iou": JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=self.ignore_index).to(device),
            "mean_dice": Dice(average='micro', num_classes=num_classes, ignore_index=self.ignore_index).to(device)
        }
        super().__init__(self.metrics)

    def update(self, predicted: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with reshaped predictions and ground truths.

        Args:
            predicted (torch.Tensor): Predicted labels (shape: [batch_size, height, width]).
            targets (torch.Tensor): Ground truth labels (shape: [batch_size, height, width]).
        """
        predicted = predicted.view(-1)
        targets = targets.view(-1)

        super().update(predicted, targets)

    def add_tests_metrics(self, device):
        """
        Add additional test-specific metrics such as accuracy, precision, and recall.
        """
        test_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index).to(device),
            "precision": Precision(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index).to(device),
            "recall": Recall(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index).to(device),
        }
        self.add_metrics(test_metrics)


def compute_class_weights(train_dataloader, num_classes, mask_key="labels", normalize="balanced", ignore_index=None):
    """
    Compute class weights based on the frequencies of each class in the dataset.

    Args:
        train_dataloader: PyTorch DataLoader containing the dataset.
        num_classes (int): Total number of classes.
        mask_key (str): Key to access masks/labels in the dataloader's batch.
        normalize (str): "max" | "sum" | "balanced" Whether to normalize the weights.
        ignore_index (int, optional): Class index to ignore in weight computation.

    Returns:
        torch.Tensor: Computed class weights of shape (num_classes,).
    """
    class_counts = torch.zeros(num_classes, dtype=torch.int64)

    for batch in tqdm(train_dataloader, desc="Compute class weights"):
        masks = batch[mask_key]
        masks = masks.view(-1)

        if ignore_index is not None:
            masks = masks[masks != ignore_index]

        counts = torch.bincount(masks, minlength=num_classes)
        class_counts += counts

    total_pixels = class_counts.sum().item()
    class_weights = total_pixels / (class_counts.float() + 1e-6)
    valid_weights = class_weights.clone()
    if normalize == "sum":
        if ignore_index is not None:
            # Exclude ignore_index from normalization
            valid_weights[ignore_index] = 0
        class_weights = valid_weights / valid_weights.sum()
    elif normalize == "max":
        if ignore_index is not None:
            # Exclude ignore_index from normalization
            valid_weights[ignore_index] = 0
        class_weights = valid_weights / valid_weights.max()
    elif normalize == "balanced":

        if ignore_index is not None:
            # Exclude ignore_index from normalization
            valid_weights[ignore_index] = 0
        class_weights = (valid_weights / valid_weights.sum()
                         ) * (num_classes - 1)

    # Ensure the weight for ignore_index is explicitly set to 0
    if ignore_index is not None:
        class_weights[ignore_index] = 0
    return class_weights.to(torch.float32)
