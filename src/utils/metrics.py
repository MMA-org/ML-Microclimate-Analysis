# util/metric.py
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch
from torchmetrics import JaccardIndex, Dice
import numpy as np
from collections import Counter


class Metrics:
    """
    A utility class to handle metrics for segmentation tasks.
    Provides functionality for IoU (Jaccard Index) and Dice coefficient calculation.
    """

    def __init__(self, num_classes):
        """
        Initialize metrics.

        Args:
            num_classes (int): Number of classes in the segmentation task.
        """
        self.num_classes = num_classes
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)
        self.dice = Dice(average='micro', num_classes=num_classes)

    def update(self, predicted, targets):
        """
        Update the metrics with predictions and ground truths.

        Args:
            predicted (torch.Tensor): Predicted labels (shape: [batch_size, height, width]).
            targets (torch.Tensor): Ground truth labels (shape: [batch_size, height, width]).
        """
        predicted, targets = predicted.view(-1), targets.view(-1)
        self.iou(predicted, targets)
        self.dice(predicted, targets)

    def compute(self):
        """
        Compute the current metric values.

        Returns:
            dict: A dictionary containing IoU and Dice scores.
        """
        return {
            "mean_iou": self.iou.compute(),
            "mean_dice": self.dice.compute(),
        }

    def reset(self):
        """
        Reset all metrics.
        """
        self.iou.reset()
        self.dice.reset()


def compute_class_weights(train_dataloader, num_classes, mask_key="labels"):
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
    return torch.tensor(class_weights, dtype=torch.float)
