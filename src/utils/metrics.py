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
            reduction='none', ignore_index=self.ignore_index, weight=self.alpha)

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

        # Apply Focal Loss Dynamics
        focal_loss = (1 - pt) ** self.gamma * ce_loss

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


def compute_class_weights(train_dataloader, num_classes, mask_key="labels", normalize=True):
    """
    Compute class weights for imbalanced datasets using a DataLoader.

    Args:
        train_dataloader: A PyTorch DataLoader that yields batches of data.
        num_classes (int): Number of classes in the dataset.
        mask_key (str): Key for the segmentation masks in the dataset.

    Returns:
        torch.Tensor: Class weights for each class.

    Raises:
        ValueError: If the dataset is empty.
        KeyError: If mask_key on invalid key.
    """

    # Check if the dataset is empty
    if len(train_dataloader.dataset) == 0:
        raise ValueError("The dataset is empty. Cannot compute class weights.")

    # Initialize a numpy array for counting occurrences
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # Iterate through the DataLoader
    for batch in tqdm(train_dataloader, desc="Compute class weights"):
        masks = batch[mask_key]  # Assuming batch is a dictionary-like object

        # Convert masks to numpy arrays if needed
        masks = masks.view(-1).cpu().numpy()

        # Update class counts
        class_counts += np.bincount(masks, minlength=num_classes)

    # Calculate the total number of pixels in the dataset (total samples)
    total_pixels = class_counts.sum()

    # Compute class weights based on the total number of pixels and the class frequencies
    class_weights = total_pixels / \
        (num_classes * class_counts + 1e-6)  # Avoid division by zero

    # Normalize class weights if needed
    if normalize:
        class_weights = class_weights / class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float)
