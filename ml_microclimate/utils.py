import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2

# Define the LandCover class labels with specified colors
lc_labels = namedtuple('LandCoverClasses', ['name', 'label', 'color'])
# Real label in dataset, need to be mapped
lc_classes = [
    lc_labels('background', 1, (255, 255, 255)),  # White
    lc_labels('building', 2, (255, 0, 0)),      # Red
    lc_labels('road', 3, (255, 255, 0)),        # Yellow
    lc_labels('water', 4, (0, 0, 255)),         # Blue
    lc_labels('barren', 5, (139, 69, 19)),      # Brown
    lc_labels('forest', 6, (0, 255, 0)),        # Green
    lc_labels('agriculture', 7, (0, 255, 255)),  # Cyan
]
# Create the id2label mapping
lc_id2label = {cls.label: cls.name for cls in lc_classes}

# Convert train_id to color mapping
lc_label_id_to_color = [c.color for c in lc_classes]

# real label in dataset, need to be mapped
building_labels = namedtuple('BuildingDataset', ['name', 'label', 'color'])
building_class = [
    building_labels('background', 1, (255, 255, 255)),  # White
    building_labels('building', 255, (255, 0, 0)),      # Black
]

# Create the id2label mapping
building_id2label = {cls.label: cls.name for cls in building_class}

# Convert train_id to color mapping
building_label_id_to_color = [c.color for c in building_class]


# Function to apply color to a mask
def apply_color_map(mask, classes):
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    unique_values = np.unique(mask)
    for label in classes:
        if label.label in unique_values:
            mask_color[mask == label.label] = label.color
    return mask_color

# Define the function to visualize the real image, ground truth, and prediction


def view_predict(image_path, mask_path, prediction_path, classes):
    # Read the image, mask, and prediction
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)

    # Raise an error if any of the files are not loaded
    if image is None:
        raise FileNotFoundError(
            f"Error: Unable to read the image at {image_path}")
    if mask is None:
        raise FileNotFoundError(
            f"Error: Unable to read the mask image at {mask_path}")
    if prediction is None:
        raise FileNotFoundError(
            f"Error: Unable to read the prediction image at {prediction_path}")

    # Get unique values in the mask and prediction
    unique_mask_values = np.unique(mask)
    unique_prediction_values = np.unique(prediction)

    # Apply the color mapping to the ground truth mask and prediction
    mask_color = apply_color_map(mask, classes)
    prediction_color = apply_color_map(prediction, classes)

    # Display the real image, ground truth, and prediction side by side
    plt.figure(figsize=(16, 8))

    # Real Image
    plt.subplot(1, 3, 1)
    # Convert BGR to RGB for display
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Real Image")
    plt.axis('off')

    # Ground Truth (Mask)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_color)
    plt.title("Ground Truth")
    plt.axis('off')

    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(prediction_color)
    plt.title("Prediction")
    plt.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Function to reduce labels in a dataset


def reduce_labels(dataset):
    """
    Apply the 'reduce labels by 1' transformation to the entire dataset.

    Args:
        dataset (Dataset): The dataset containing image and mask pairs.

    Returns:
        transformed_data (list): A list of dictionaries with transformed images and masks.
    """
    transformed_data = []

    for data in dataset:
        image = data['image']
        mask = np.array(data['mask'])  # Convert mask to numpy array

        # Reduce mask labels by 1
        reduced_mask = np.copy(mask)
        reduced_mask[mask > 0] -= 1  # Reduce all valid classes by 1

        # Store the transformed data
        transformed_data.append({'image': image, 'mask': reduced_mask})

    return transformed_data

# Function to map 255 to 0 in a binary dataset


def map_binary_dataset(dataset):
    """
    Apply the 'map 255 to 0' transformation to the entire dataset.

    Args:
        dataset (Dataset): The dataset containing image and mask pairs.

    Returns:
        transformed_data (list): A list of dictionaries with transformed images and masks.
    """
    transformed_data = []

    for data in dataset:
        image = data['image']
        mask = np.array(data['mask'])  # Convert mask to numpy array

        # Map 255 to 0 in the mask
        mapped_mask = np.copy(mask)
        mapped_mask[mapped_mask == 255] = 0  # Map 255 to 0

        # Store the transformed data
        transformed_data.append({'image': image, 'mask': mapped_mask})

    return transformed_data
