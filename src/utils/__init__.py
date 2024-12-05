"""
This module provides functionality for loading configuration files and defining land cover class labels.

Modules:
    yaml: YAML parser and emitter for Python.
    collections.namedtuple: Factory function for creating tuple subclasses with named fields.
    matplotlib.pyplot: A state-based interface to matplotlib, a comprehensive library for creating static, animated, and interactive visualizations in Python.
    numpy: A fundamental package for scientific computing with Python.
    PIL.Image: A module from the Python Imaging Library (PIL) for opening, manipulating, and saving many different image file formats.

Functions:
    load_config(config_path='config.yaml'):
        Reads in the configuration file from the specified path and returns the configuration as a dictionary.

Named Tuples:
    LandCoverClasses:
        A named tuple to define land cover class labels with specified colors.

Variables:
    lc_labels: A named tuple 'LandCoverClasses' with fields 'label', 'id', and 'color'.
    lc_classes: A list of 'LandCoverClasses' named tuples representing different land cover types with their corresponding labels, IDs, and colors.
"""
import yaml
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_config(config_path='config.yaml'):
    """
    Reads in the configuration file from the specified path and returns the configuration as a dictionary.

    Args:
        config_path (str): The path to the configuration file. Default is 'config.yaml'.

    Returns:
        dict: The configuration data.
    """
    with open(config_path) as p:
        config = yaml.safe_load(p)
    return config


# Define the LandCover class labels with specified colors
lc_labels = namedtuple('LandCoverClasses', ['label', 'id', 'color'])
# Real label in dataset, need to be mapped
lc_classes = [
    lc_labels('background', 0, (255, 255, 255)),  # White
    lc_labels('building', 1, (255, 0, 0)),      # Red
    lc_labels('road', 2, (255, 255, 0)),        # Yellow
    lc_labels('water', 3, (0, 0, 255)),         # Blue
    lc_labels('barren', 4, (139, 69, 19)),      # Brown
    lc_labels('woodland', 5, (0, 255, 0)),        # Green
]
# Create the id2label mapping
lc_id2label = {cls.id: cls.label for cls in lc_classes}
# Convert id to color mapping
lc_id2color = {cls.id: cls.color for cls in lc_classes}


def apply_color_map(mask: np.ndarray) -> np.ndarray:
    """
    Map class indices to RGB values for visualization.

    Args:
        mask (np.ndarray): 2D array of class indices (shape: [height, width]).

    Returns:
        np.ndarray: 3D array of RGB values (shape: [height, width, 3]).
    """

    # Initialize an empty RGB image
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply color mapping
    for class_id, color in lc_id2color.items():
        color_mask[mask == class_id] = color

    return color_mask


def plot_image_and_mask(image_path: str, mask: np.ndarray):
    """
    Plots an image and its corresponding mask side by side.

    Args:
        image_path (str): The path to the image file.
        mask (np.ndarray): The mask array to be applied to the image.

    Returns:
        None
    """
    # Open the image
    image = Image.open(image_path)

    # Apply color map to the mask
    color_mask = apply_color_map(mask)

    # Plot the image and mask side by side
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')

    # Plot the mask with color map
    ax[1].imshow(color_mask)
    ax[1].set_title("Mask")
    ax[1].axis('off')

    plt.show()
