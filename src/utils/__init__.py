"""
Utilities for Configuration Management, Visualization, and Land Cover Classification.

Modules:
    - Configuration loading and management.
    - Confusion matrix generation and visualization.
    - Land cover label and color mapping.
"""

from pathlib import Path
import yaml
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Config:
    """
    Encapsulates configuration data, enabling nested attribute-based access.
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    def __getattr__(self, name):
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config.from_dict(value)
        elif value is not None:
            return value
        raise AttributeError(f"Configuration key '{name}' not found.")

    @staticmethod
    def from_dict(config_dict):
        """Create a Config object from a dictionary."""
        config = Config.__new__(
            Config)  # Create a new instance without calling __init__
        config._config = config_dict
        return config

    def get(self, *keys, default=None):
        """
        Get nested configuration values with a fallback default.
        """
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default


def find_checkpoint(config: Config, version: str) -> str:
    """
    Locate the single checkpoint file in the specified versioned directory.

    Args:
        config (Config): Configuration object.
        version (str): The version folder name (e.g., "version_0").

    Returns:
        str: Path to the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint file is found.
    """
    checkpoint_dir = Path(config.project.log_dir) / version / "checkpoints"
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found in directory: {checkpoint_dir}")

    return str(checkpoint_files[0])


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """
    Generate and save a confusion matrix plot.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of class labels.
        save_path (str or Path, optional): Path to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")

    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


# Define land cover class labels with associated colors
LandCoverClass = namedtuple('LandCoverClass', ['label', 'id', 'color'])
lc_classes = [
    LandCoverClass('background', 0, (255, 255, 255)),  # White
    LandCoverClass('building', 1, (255, 0, 0)),       # Red
    LandCoverClass('road', 2, (255, 255, 0)),         # Yellow
    LandCoverClass('water', 3, (0, 0, 255)),          # Blue
    LandCoverClass('barren', 4, (139, 69, 19)),       # Brown
    LandCoverClass('woodland', 5, (0, 255, 0)),       # Green
]

# Create mappings for label-to-id and id-to-color
lc_id2label = {cls.id: cls.label for cls in lc_classes}
lc_id2color = {cls.id: cls.color for cls in lc_classes}


def apply_color_map(mask: np.ndarray) -> np.ndarray:
    """
    Map class indices to RGB values for visualization.

    Args:
        mask (np.ndarray): 2D array of class indices.

    Returns:
        np.ndarray: 3D array of RGB values.
    """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in lc_id2color.items():
        color_mask[mask == class_id] = color

    return color_mask


def plot_image_and_mask(image_path: str, mask: np.ndarray):
    """
    Display an image and its corresponding mask side by side.

    Args:
        image_path (str): Path to the image file.
        mask (np.ndarray): 2D array representing the mask.

    Returns:
        None
    """
    image = Image.open(image_path)
    color_mask = apply_color_map(mask)

    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')
    ax[1].imshow(color_mask)
    ax[1].set_title("Mask")
    ax[1].axis('off')
    plt.show()
