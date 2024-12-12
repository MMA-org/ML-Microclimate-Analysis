# util/metric.py
from tqdm import tqdm
from torchmetrics import MetricCollection, JaccardIndex, Dice
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for static datasets with fixed class weights.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma=2, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): Per-class weights (shape: [num_classes]).
                                      If None, no per-class weighting is applied.
            gamma (float): Focusing parameter for the focal loss (default: 2).
            reduction (str): Reduction method for the loss ('none', 'mean', 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Precomputed class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.

        Args:
            inputs (Tensor): Logits predicted by the model (shape: [batch, classes, height, width]).
            targets (Tensor): Ground truth labels (shape: [batch, height, width]).

        Returns:
            Tensor: Computed focal loss.
        """
        # Ensure alpha is on the correct device
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)

        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha)

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

    def __init__(self, num_classes, device="cpu"):
        """
        Initialize metrics.

        Args:
            num_classes (int): Number of classes in the segmentation task.
        """
        self.num_classes = num_classes
        self.metrics = {
            "mean_iou": JaccardIndex(task='multiclass', num_classes=num_classes).to(device),
            "mean_dice": Dice(average='micro', num_classes=num_classes).to(device)
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

    total_samples = class_counts.sum()

    # Compute class weights
    class_weights = total_samples / \
        (class_counts + 1e-6)  # Avoid division by zero
    if normalize:
        class_weights = class_weights / class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float)
