# util/metric.py
from tqdm import tqdm
from torchmetrics import MetricCollection, JaccardIndex, Dice, Accuracy, Precision, Recall
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for static datasets with fixed class weights.
    """

    def __init__(self, num_class, alpha: Optional[torch.Tensor] = None, gamma=2, reduction='mean', ignore_index=None):
        """
        Args:
            alpha (Tensor, optional): Per-class weights (shape: [num_classes]).
                                      If None, no per-class weighting is applied.
            gamma (float): Focusing parameter for the focal loss (default: 2).
            reduction (str): Reduction method for the loss ('none', 'mean', 'sum').
        """
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha if alpha is not None else torch.ones(
            self.num_class).float()  # Precomputed class weights
        self.gamma = gamma
        self.reduction = reduction
        self.num_class = num_class
        self.ignore_index = ignore_index if ignore_index is not None else -100

        if self.ignore_index >= 0 and self.ignore_index < self.num_class:
            self.alpha[self.ignore_index] = 0

        self.ce_loss = nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.ignore_index)

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.

        Args:
            inputs (Tensor): Logits predicted by the model (shape: [batch, classes, height, width]).
            targets (Tensor): Ground truth labels (shape: [batch, height, width]).

        Returns:
            Tensor: Computed focal loss.
        """

        # Compute Cross-Entropy Loss
        self.alpha = self.alpha.to(inputs.device)

        # Compute Cross-Entropy Loss
        ce_loss = self.ce_loss(inputs, targets)

        # Compute Probabilities
        pt = torch.exp(-ce_loss)
        alpha = self.alpha[targets]  # Shape: [batch_size, height, width]

        # Add extra dimension to alpha for broadcasting
        alpha = alpha.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
        # Apply Focal Loss Dynamics
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index).to(device),
            "precision": Precision(task="multiclass", num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index).to(device),
            "recall": Recall(task="multiclass", num_classes=self.num_classes, average="micro", ignore_index=self.ignore_index).to(device),
        }
        self.add_metrics(test_metrics)


def compute_class_weights(train_dataloader, num_classes, mask_key="labels", normalize=True, ignore_index=None):
    """
    Compute class weights for imbalanced datasets using a DataLoader.

    Args:
        train_dataloader: A PyTorch DataLoader that yields batches of data.
        num_classes (int): Number of classes in the dataset.
        mask_key (str): Key for the segmentation masks in the dataset.
        ignore_index (int, optional): The index of the class to ignore during weight computation.

    Returns:
        torch.Tensor: Class weights for each class.
    """

    # Initialize class counts (excluding ignore_index if specified)
    class_counts = np.zeros(num_classes, dtype=np.float64)

    # Iterate through the DataLoader
    for batch in tqdm(train_dataloader, desc="Compute class weights"):
        masks = batch[mask_key].view(-1).cpu().numpy()

        # If ignore_index is specified, exclude it from class counts
        if ignore_index is not None:
            masks = masks[masks != ignore_index]

        # Update class counts
        class_counts += np.bincount(masks, minlength=num_classes)

    # Compute class weights (inverse of class frequencies)
    class_weights = 1 / (class_counts + 1e-6)

    # Set weight for ignore_index class to 0
    if ignore_index is not None:
        class_weights[ignore_index] = 0

    # Normalize class weights, excluding ignore_index class from normalization
    if normalize:
        # Normalize only over the valid classes (excluding ignore_index)
        # Filter out zero weights
        valid_class_weights = class_weights[class_weights > 0]
        class_weights[class_weights > 0] = valid_class_weights / \
            valid_class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float)
