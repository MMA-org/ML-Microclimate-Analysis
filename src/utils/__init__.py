
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import json
from .config import Config

# Define land cover class labels with associated colors
LandCoverClass = namedtuple('LandCoverClass', ['label', 'id', 'color'])
lc_classes = [
    LandCoverClass('background', 0, (0, 0, 0)),                  # Black
    LandCoverClass('forest', 1, (44, 160, 44)),                 # Green
    LandCoverClass('water', 2, (31, 119, 180)),                 # Blue
    LandCoverClass('agricultural', 3, (140, 86, 75)),           # Brown
    LandCoverClass('residential,commercial,industrial',
                   4, (127, 127, 127)),  # Gray
    LandCoverClass('grassland,swamp,shrubbery', 5, (188, 189, 34)),   # Olive
    LandCoverClass('railway,trainstation', 6, (255, 127, 14)),        # Orange
    LandCoverClass('railway,highway,squares', 7, (148, 103, 189)),    # Purple
    LandCoverClass('airport,shipyard', 8, (23, 190, 207)),            # Cyan
    LandCoverClass('roads', 9, (214, 39, 40)),                       # Red
    LandCoverClass('buildings', 10, (227, 119, 194))                 # Pink
]

# Create mappings for label-to-id and id-to-color
lc_id2label = {cls.id: cls.label for cls in lc_classes}
lc_id2color = {cls.id: cls.color for cls in lc_classes}


def get_next_version(logs_dir: Path) -> str:
    """
    Get the next version number for the logs directory containing only 'version_*' folders.

    Args:
        logs_dir (Path): The base directory where 'lightning_logs' are stored.

    Returns:
        str: The next version number in the format 'version_{n}'.
    """
    lightning_logs_dir = logs_dir / "lightning_logs"

    # Ensure the 'lightning_logs' directory exists
    if not lightning_logs_dir.exists():
        return "version_0"

    # Extract version numbers from folder names
    version_numbers = [
        int(d.name.split("_")[1])
        for d in lightning_logs_dir.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]

    # Determine the next version number
    next_version = max(version_numbers, default=-1) + 1
    return f"version_{next_version}"


def find_checkpoint(config, version: str) -> Path:
    """
    Locate the single checkpoint file in the specified versioned directory.

    Args:
        config: Configuration object.
        version (str): The version folder name (e.g., "version_0").

    Returns:
        Path: Absolute path to the checkpoint file.

    Raises:
        CheckpointDirectoryError: If the checkpoint directory is missing or invalid.
        CheckpointNotFoundError: If no checkpoint file is found.
        MultipleCheckpointsError: If multiple checkpoint files are found.
    """
    from .errors import CheckpointNotFoundError, CheckpointDirectoryError, MultipleCheckpointsError
    checkpoint_dir = Path(config.project.logs_dir) / \
        "checkpoints" / f"version_{version}"

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


def save_confusion_matrix_plot(y_true, y_pred, labels, save_path, metrics=None, title="Confusion Matrix"):
    """
    Save a confusion matrix plot to a file.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of class labels.
        save_path (str or Path): Path to save the confusion matrix plot.
        metrics (dict, optional): Dictionary of metrics to annotate below the confusion matrix.
        title (str, optional): Title for the confusion matrix plot.
    """
    # Ensure inputs are numpy arrays and flatten them
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

    # Plot confusion matrix using ConfusionMatrixDisplay and apply LogNorm normalization
    fig, ax = plt.subplots(figsize=(12, 12))
    norm = mcolors.LogNorm(vmin=cm.min(), vmax=cm.max(), clip=False)

    # Plot the confusion matrix manually using `imshow` and LogNorm
    cax = ax.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest", norm=norm)

    # Add color bar with label indicating the use of log scale
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Log Scale Values', rotation=270, labelpad=15)

    # Add title and labels
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Annotate each cell with the numeric value, displayed in log scale
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            # Apply log transform (skip log(0))
            log_value = np.log10(value) if value > 0 else 0
            ax.text(j, i, f'{log_value:.2f}', ha="center",
                    va="center", color="white", fontsize=12)

    # Add text annotation for metrics below the confusion matrix
    if metrics:
        metric_items = list(metrics.items())
        rows = [
            "    " +
            "    ".join([f"{key}: {value:.3f}" for key,
                        value in metric_items[i:i + 3]])
            for i in range(0, len(metric_items), 3)
        ]
        metrics_text = "\n".join(rows)
        ax.text(
            0.5, -0.15,  # Adjust position
            metrics_text,
            ha='center', va='top', fontsize=12, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray',
                      facecolor='white', alpha=0.5)
        )

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_class_weights(weights_file):
    """
    Load precomputed class weights from a file and return as a torch tensor.

    Args:
        weights_file (Path): Path to the weights file.

    Returns:
        torch.Tensor: Loaded class weights as a tensor.
    """
    import torch
    print("Loading precomputed class weights from file.")
    with weights_file.open("r") as f:
        return torch.tensor(json.load(f), dtype=torch.float)


def save_class_weights(weights_file, class_weights):
    """
    Save computed class weights to a file.

    Args:
        weights_file (Path): Path to the weights file.
        class_weights (list): Class weights to save.
    """
    print("Saving class weights to file.")
    with weights_file.open("w") as f:
        json.dump(class_weights.tolist(), f)


def apply_color_map(mask, id2color) -> np.ndarray:
    """
    Map class indices to RGB values for visualization.

    Args:
        mask (Image.Image): 2D image with class indices (PIL Image).
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
        image_path (str): Path to the image file.
        mask (np.ndarray): 2D array representing the mask.

    Returns:
        None
    """
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    color_mask = apply_color_map(mask, id2color)

    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')
    ax[1].imshow(color_mask)
    ax[1].set_title("Mask")
    ax[1].axis('off')
    plt.show()
