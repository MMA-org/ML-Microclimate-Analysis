
from pathlib import Path
import yaml
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import torch

# Define land cover class labels with associated colors
LandCoverClass = namedtuple('LandCoverClass', ['label', 'id', 'color'])
lc_classes = [
    LandCoverClass('background', 0, (255, 255, 255)),  # White
    LandCoverClass('building', 1, (255, 0, 0)),       # Red
    LandCoverClass('road', 2, (255, 255, 0)),         # Yellow
    LandCoverClass('water', 3, (0, 0, 255)),          # Blue
    LandCoverClass('barren', 4, (139, 69, 19)),       # Brown
    LandCoverClass('woodland', 5, (0, 255, 0)),       # Green
    LandCoverClass('agriculture', 6, (50, 143, 168)),  # Purple
]

# Create mappings for label-to-id and id-to-color
lc_id2label = {cls.id: cls.label for cls in lc_classes}
lc_id2color = {cls.id: cls.color for cls in lc_classes}


class Config:
    """
    Encapsulates configuration data, enabling nested attribute-based access.
    Automatically creates directories specified in the `project` section.
    """

    def __init__(self, config_path="config.yaml", create_dirs=True):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        if create_dirs:
            self._create_directories()

    def __getattr__(self, name):
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config.from_dict(value)
        elif value is not None:
            return value
        raise AttributeError(f"Configuration key '{name}' not found.")

    @staticmethod
    def from_dict(config_dict, create_dirs=True):
        """Create a Config object from a dictionary."""
        config = Config.__new__(
            Config)  # Create a new instance without calling __init__
        config._config = config_dict
        if create_dirs:
            config._create_directories()
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

    def _create_directories(self):
        """
        Automatically create directories specified in the `project` section.
        """
        project_config = self._config.get("project", {})
        created_dirs = []  # Collect created directory paths

        for key, dir_path in project_config.items():
            dir_path = Path(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))  # Add to list

        if created_dirs:
            print(f"Directories created: {' '.join(created_dirs)}")


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
    checkpoint_dir = Path(config.project.logs_dir) / \
        "checkpoints" / f"version_{version}"
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found in directory: {checkpoint_dir}")

    return str(checkpoint_files[0])


def save_confusion_matrix_plot(y_true, y_pred, labels, save_path):
    """
    Save a confusion matrix plot to a file.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of class labels.
        save_path (str or Path): Path to save the confusion matrix plot.
    """
    if isinstance(y_true, list):
        y_true = np.concatenate(y_true).flatten()
    else:
        y_true = np.array(y_true).flatten()

    if isinstance(y_pred, list):
        y_pred = np.concatenate(y_pred).flatten()
    else:
        y_pred = np.array(y_pred).flatten()

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def load_class_weights(weights_file):
    """
    Load precomputed class weights from a file and return as a torch tensor.

    Args:
        weights_file (Path): Path to the weights file.

    Returns:
        torch.Tensor: Loaded class weights as a tensor.
    """
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
