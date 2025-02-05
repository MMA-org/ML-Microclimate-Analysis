from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def get_last_version(logs_dir: Path) -> int:
    """
    Get the last version number for the logs directory containing only 'version_[n]' folders.

    Args:
        logs_dir (Path): The base directory where 'lightning_logs' are stored.

    Returns:
        int: The last version number. Returns -1 if no 'version_[n]' folders exist.
    """
    lightning_logs_dir = logs_dir / "lightning_logs"

    # Ensure the 'lightning_logs' directory exists
    if not lightning_logs_dir.exists():
        return -1

    # Extract version numbers from folder names
    version_numbers = [
        int(d.name.split("_")[1])
        for d in lightning_logs_dir.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]

    return max(version_numbers, default=-1)


def get_next_version(logs_dir: Path) -> str:
    """
    Get the next version number for the logs directory containing only 'version_[n]' folders.

    Args:
        logs_dir (Path): The base directory where 'lightning_logs' are stored.

    Returns:
        str: The next version number in the format 'version_[n]'.
    """
    last_version = get_last_version(logs_dir)
    next_version = last_version + 1
    return f"version_{next_version}"


def find_checkpoint(checkpoints_dir, version: int) -> Path:
    """
    Locate the single checkpoint file in the specified versioned directory.

    Args:
        chekpoints_dir (str): A path to parent checkpoints directory.
        version (int): The version folder name (e.g., "version_0").

    Returns:
        Path: Absolute path to the checkpoint file.

    Raises:
        CheckpointDirectoryError: If the checkpoint directory is missing or invalid.
        CheckpointNotFoundError: If no checkpoint file is found.
        MultipleCheckpointsError: If multiple checkpoint files are found.
    """
    from ucs.core.errors import (
        CheckpointDirectoryError,
        CheckpointNotFoundError,
        MultipleCheckpointsError,
    )

    checkpoint_dir = Path(checkpoints_dir) / f"version_{version}"

    # Check if the checkpoint directory exists
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise CheckpointDirectoryError(checkpoint_dir)

    # Locate checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        raise CheckpointNotFoundError(checkpoint_dir)

    if len(checkpoint_files) > 1:
        raise MultipleCheckpointsError(checkpoint_dir)

    return checkpoint_files[0].resolve()


def format_metrics(metrics):
    """
    Format metrics for display below the confusion matrix.

    Args:
        metrics (dict): Dictionary containing metric names and values.

    Returns:
        str: Formatted string representation of the metrics for plot box.
    """
    metric_items = list(metrics.items())
    rows = [
        "    ".join([f"{key}: {value:.3f}" for key, value in metric_items[i : i + 3]])
        for i in range(0, len(metric_items), 3)
    ]
    return "\n".join(rows)


def save_confusion_matrix_plot(
    conf_matrix, labels, save_path, metrics=None, title="Confusion Matrix"
):
    """
    Save a confusion matrix plot to a file using sklearn's ConfusionMatrixDisplay.

    Args:
        conf_matrix (np.ndarray): Confusion matrix containing `y_true` (actual labels) and `y_pred` (predicted labels).
        labels (list): List of class labels.
        save_path (Path): path to save the confusion matrix plot.
        metrics (dict, optional): Dictionary of metrics to annotate below the confusion matrix. Defaults to None.
        title (str, optional): Title for the confusion matrix plot. Defaults to "Confusion Matrix".
    """

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)

    # Plot the confusion matrix
    _, ax = plt.subplots(figsize=(12, 12))
    # 'd' for integer display
    disp.plot(cmap="Blues", ax=ax, values_format="d")

    # Add title
    ax.set_title(title)
    bbox_params = {
        "boxstyle": "round,pad=0.3",
        "edgecolor": "gray",
        "facecolor": "white",
        "alpha": 0.5,
    }
    # Add text annotation for metrics below the confusion matrix
    if metrics:
        metrics_text = format_metrics(metrics)
        ax.text(
            0.5,
            -0.15,
            metrics_text,
            ha="center",
            va="top",
            fontsize=12,
            transform=ax.transAxes,
            bbox=bbox_params,
        )

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_class_weights(class_weights_dir):
    """
    Load precomputed class weights from a file and return as a torch tensor.

    Args:
        class_weights_dir (str): path to the dir contain weights file.

    Returns:
        torch.Tensor: Loaded class weights as a tensor.
    """
    from torch import load

    weights_file = (Path(class_weights_dir) / "class_weights.pt").resolve()
    if weights_file.exists():
        print("Loading precomputed class weights from file.")
        return load(str(weights_file), weights_only=True)
    return None


def save_class_weights(class_weights_dir, class_weights):
    """
    Save computed class weights to a file.

    Args:
        class_weights_dir (str): path to the dir contain weights file.
        class_weights (list): Class weights to save.
    """
    from torch import save

    weights_file = (Path(class_weights_dir) / "class_weights.pt").resolve()
    print("Saving class weights to file.")
    save(class_weights, str(weights_file))


def apply_color_map(mask, id2color) -> np.ndarray:
    """
    Map class indices to RGB values for visualization.

    Args:
        mask (np.ndarray): 2D array with class indices.
        id2color (dict): Dictionary mapping class IDs to RGB color tuples.

    Returns:
        np.ndarray: 3D array of RGB values.
    """
    # Convert PIL Image to NumPy array
    mask = np.array(mask)

    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in id2color.items():
        color_mask[mask == class_id] = color

    return color_mask


def plot_image_and_mask(image, mask: np.ndarray, id2color):
    """
    Display an image and its corresponding mask side by side.

    Args:
        image (PIL.Image.Image or str): Image to display, or the path to the image file.
        mask (np.ndarray): 2D array representing the mask.
        id2color (dict): Dictionary mapping class IDs to RGB color tuples.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    color_mask = apply_color_map(mask, id2color)

    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].imshow(color_mask)
    ax[1].set_title("Mask")
    ax[1].axis("off")
    plt.show()
