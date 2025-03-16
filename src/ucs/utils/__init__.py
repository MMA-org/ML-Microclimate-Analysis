from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_last_version(logs_dir: Path, project_name: str = "lightning_logs") -> int:
    """
    Get the last version number for the logs directory containing only 'version_[n]' folders.

    Args:
        logs_dir (Path): The base directory where 'lightning_logs' are stored.

    Returns:
        int: The last version number. Returns -1 if no 'version_[n]' folders exist.
    """
    lightning_logs_dir = Path(logs_dir) / project_name

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


def get_next_version(logs_dir: Path, project_name: str) -> str:
    """
    Get the next version number for the logs directory containing only 'version_[n]' folders.

    Args:
        logs_dir (Path): The base directory where 'lightning_logs' are stored.

    Returns:
        str: The next version number in the format 'version_[n]'.
    """
    last_version = get_last_version(logs_dir, project_name)
    next_version = last_version + 1
    return next_version


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
    confmat, save_path, metrics=None, title="Confusion Matrix", labels=None
):
    """
    Save a confusion matrix plot using `torchmetrics`' built-in `plot()` method.

    Args:
        confmat (MulticlassConfusionMatrix): TorchMetrics confusion matrix instance.
        save_path (str or Path): Path to save the confusion matrix plot.
        metrics (dict, optional): Dictionary of metrics to annotate below the confusion matrix.
        title (str, optional): Title for the confusion matrix plot.
    """
    # Compute confusion matrix
    _, ax = plt.subplots(figsize=(8, 6))
    confmat.plot(ax=ax, cmap="Blues", labels=labels)

    # Set title
    ax.set_title(title)

    # Add text annotation for additional metrics
    if metrics:
        metrics_text = "\n".join(
            [f"{key}: {value:.4f}" for key, value in metrics.items()]
        )
        bbox_params = {
            "boxstyle": "round,pad=0.3",
            "edgecolor": "gray",
            "facecolor": "white",
            "alpha": 0.5,
        }
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
    save_path = Path(save_path)
    plt.savefig(save_path)
    plt.close()


def load_class_weights(class_weights_dir, strategy):
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
        loaded_data = load(str(weights_file), weights_only=True)
        if loaded_data["strategy"] != strategy:
            return None
    return loaded_data['class_weights']


def save_class_weights(class_weights_dir, class_weights, strategy):
    """
    Save computed class weights to a file.

    Args:
        class_weights_dir (str): path to the dir contain weights file.
        class_weights (list): Class weights to save.
    """
    from torch import save

    weights_file = (Path(class_weights_dir) / "class_weights.pt").resolve()
    data_to_save = {'class_weights': class_weights, 'strategy': strategy}
    print(f"Saving class weights to {weights_file} with strategy: {strategy}")
    save(data_to_save, str(weights_file))


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
