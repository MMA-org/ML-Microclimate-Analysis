# util/metric.py
from typing import Optional, Literal
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchmetrics import MetricCollection, JaccardIndex, Dice, Accuracy, Precision, Recall
from core.errors import NormalizeError, LossWeightsSizeError, LossWeightsTypeError


# CeDiceLoss class
class CeDiceLoss(nn.Module):
    """
    A combined loss function that incorporates Cross-Entropy Loss (CE) and Dice Loss
    for semantic segmentation tasks. The CE loss accounts for class imbalance, 
    while the Dice loss measures the overlap between predicted and ground truth masks, 
    providing a robust metric for segmentation accuracy.

    Args:
        num_classes (int): The total number of classes in the segmentation task.
        alpha (float, optional): Weight for Cross-Entropy Loss in the combined loss. 
            Default is 0.8.
        beta (float, optional): Weight for Dice Loss in the combined loss. Default is 0.2.
        weights (list, np.ndarray, torch.Tensor, optional): Class weights for 
            handling imbalanced datasets. If provided, it must match the number of classes.
        ignore_index (int, optional): Class index to ignore during loss computation. 
            Pixels with this index are excluded from both CE and Dice loss calculations. 
            Default is None.
        reduction (str, optional): Specifies the reduction to apply to the output of 
            the CE loss. Must be one of "none", "mean" (default), or "sum".

    Attributes:
        alpha (float): Weight for the CE loss component.
        beta (float): Weight for the Dice loss component.
        num_classes (int): Number of classes in the segmentation task.
        ignore_index (int or None): Class index to ignore during loss computation.
        weights (torch.Tensor or None): Tensor containing class weights for CE loss.
        ce_loss (torch.nn.CrossEntropyLoss): Cross-Entropy loss module.
        dice_score (Dice): Dice score metric for multi-class segmentation.
    """

    def __init__(self, num_classes, alpha=0.8, beta=0.2, weights=None, ignore_index=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.weights = self.initialize_weights(weights, num_classes)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100,
            reduction=reduction,
            weight=self.weights,
        )
        self.dice_score = Dice(
            average="macro",
            multiclass=True,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
        )

    def forward(self, inputs, targets):
        """
        Forward pass of the loss computation that combines Cross-Entropy loss and Dice loss.

        This method computes the total loss by combining the Cross-Entropy loss (CE loss) and Dice loss.
        The Cross-Entropy loss is calculated using the `inputs` and `targets`, while the Dice loss is
        calculated by first applying a softmax function to the `inputs` to get the predicted probabilities.
        The Dice loss is computed by comparing the predicted probabilities with the target labels.

        The total loss is a weighted sum of the CE loss and Dice loss, with the weights specified by 
        the `alpha` and `beta` parameters.

        Args:
            inputs (torch.Tensor): The model's raw output logits (before softmax).
                                Shape: (batch_size, num_classes, height, width) for 2D inputs.
            targets (torch.Tensor): The ground truth labels for comparison.
                                Shape: (batch_size, height, width) for 2D inputs.

        Returns:
            torch.Tensor: The weighted sum of the Cross-Entropy loss and Dice loss.
                        A scalar value representing the total loss.

        """
        ce_loss = self.ce_loss(inputs, targets)
        probs = torch.softmax(inputs, dim=1)
        dice_loss = 1 - self.dice_score(probs, targets)
        return self.alpha * ce_loss + self.beta * dice_loss

    def initialize_weights(self, weights, num_classes):
        """
        Initialize and validate weights for a loss function.

        Args:
            weights (list, np.ndarray, torch.Tensor, optional): Weighting factor(s) for each class. 
            num_classes (int): The total number of classes for the loss function.

        Returns:
            torch.Tensor: A tensor of weights for all classes.

        Raises:
            LossWeightsTypeError: If the `weights` argument is of an unsupported type.
            LossWeightsSizeError: If the size of `weights` does not match `num_classes`.
        """

        if weights is None:
            return None  # No weights provided, return None

        # Convert weights to tensor
        if isinstance(weights, (list, np.ndarray)):
            weights_tensor = torch.tensor(weights, dtype=torch.float)
        elif isinstance(weights, torch.Tensor):
            weights_tensor = weights
        else:
            raise LossWeightsTypeError(type(weights))
        if weights_tensor.size(0) != num_classes:
            raise LossWeightsSizeError(
                weights_tensor.size(0), num_classes)

        return weights_tensor


class FocalLoss(nn.CrossEntropyLoss):
    """
        Focal Loss for addressing class imbalance in classification tasks.

        Args:
            num_classes (int): Number of classes.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            alpha (float, list, np.ndarray, torch.Tensor, optional): Weighting factor for each class. Defaults to None.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to None.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to 'mean'.

        Attributes:
            ignore_index (int): The index to ignore in the target.
            gamma (float): The focusing parameter.
            reduction (str): The reduction method to apply to the output.
            num_classes (int): The number of classes.
            alpha (torch.Tensor): The weighting factor for each class.

        Raises:
            FocalAlphaTypeError: If the alpha type is unsupported.
            FocalAlphaSizeError: If alpha does not match `num_classes`.
    """

    def __init__(self, num_classes, gamma=2.0, alpha=None, ignore_index=None, reduction='mean'):
        self.ignore_index = ignore_index if ignore_index is not None else -100
        super().__init__(ignore_index=self.ignore_index, reduction='none')
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.alpha = self.__set_alpha__(alpha)

    def forward(self, inputs, target):
        self.alpha = self.alpha.to(inputs.device)
        cross_entropy = super().forward(inputs, target)
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

    def __set_alpha__(self, alpha):
        """
        Set the alpha value for class weighting.

        Args:
            alpha (float, list, np.ndarray, torch.Tensor, optional): Weighting factor for each class.

        Returns:
            torch.Tensor: The alpha tensor.

        Raises:
            FocalAlphaTypeError: If the alpha type is unsupported.
            FocalAlphaSizeError: alpha does not match num_classes.
        """
        if alpha is None:
            alpha_tensor = torch.ones(self.num_classes, dtype=torch.float)
        elif isinstance(alpha, (float, int)):
            alpha_tensor = torch.full(
                (self.num_classes,), alpha, dtype=torch.float)
        elif isinstance(alpha, (np.ndarray, list)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha
        else:
            raise LossWeightsTypeError(type(alpha))

        if alpha_tensor.size(0) != self.num_classes:
            raise LossWeightsSizeError(alpha_tensor.size(0), self.num_classes)

        return alpha_tensor


class SegMetrics(MetricCollection):
    """
    A utility class to handle metrics for segmentation tasks.
    Provides functionality for IoU (Jaccard Index) and Dice coefficient calculation.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        device (str, optional): Device to run the metrics on. Default is "cpu".
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default is None.

    Attributes:
        num_classes (int): The number of classes.
        ignore_index (int): The index to ignore in the target.
        metrics (dict): Dictionary of metrics to compute.
    """

    def __init__(self, num_classes, device="cpu", ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = {
            "mean_iou": JaccardIndex(task='multiclass', average="macro", num_classes=num_classes, ignore_index=self.ignore_index).to(device),
            "mean_dice": Dice(average='macro', num_classes=num_classes, ignore_index=self.ignore_index).to(device)
        }
        super().__init__(self.metrics)

    def update(self, predicted: torch.Tensor, targets: torch.Tensor) -> None:
        predicted = predicted.view(-1)
        targets = targets.view(-1)

        super().update(predicted, targets)


class TestMetrics(SegMetrics):
    """
    A utility class to handle metrics for segmentation tasks.
    Provides functionality for IoU (Jaccard Index) and Dice coefficient calculation.

    Args:
        num_classes (int): Number of classes in the segmentation task.
        device (str, optional): Device to run the metrics on. Default is "cpu".
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default is None.

    Attributes:
        num_classes (int): The number of classes.
        ignore_index (int): The index to ignore in the target.
        metrics (dict): Dictionary of metrics to compute.
    """

    def __init__(self, num_classes, device="cpu", ignore_index: Optional[int] = None):
        super().__init__(num_classes, device, ignore_index)
        test_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index).to(device),
            "precision": Precision(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index).to(device),
            "recall": Recall(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index).to(device),
        }
        self.add_metrics(test_metrics)


def compute_class_weights(train_dataloader, num_classes, mask_key="labels", normalize: Literal["sum", "max", "none", "balanced"] = "none", ignore_index=None):
    """
    Compute class weights based on the frequencies of each class in the dataset.

    Args:
        train_dataloader (DataLoader): PyTorch DataLoader containing the dataset.
        num_classes (int): Total number of classes.
        mask_key (str): Key to access masks/labels in the dataloader's batch.
        normalize (str): Method to normalize weights:
            - "sum": Scales weights to sum to 1.
            - "max": Scales weights so the maximum weight is 1.
            - "none": Leaves weights unnormalized.
            - "balanced": Ensures equal contribution from all classes.
        ignore_index (int, optional): Class index to ignore in weight computation.

    Returns:
        torch.Tensor: Computed class weights of shape (num_classes,).

    Raises:
        NormalizeError: If the normalization method is unsupported.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.int64)

    for batch in tqdm(train_dataloader, desc="Compute class weights"):
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
        torch.tensor(0.0, dtype=torch.float32)
    )

    # Compute class weights based on the normalization strategy
    if normalize == "sum":
        class_weights /= class_weights.sum()

    elif normalize == "max":
        class_weights /= class_weights.max()

    elif normalize == "balanced":
        effective_classes = num_classes - \
            (1 if ignore_index is not None else 0)
        class_weights = class_weights * \
            (effective_classes / class_weights.sum())

    elif normalize == "none":
        pass
    else:
        raise NormalizeError(normalize)

    return class_weights.to(torch.float32)
