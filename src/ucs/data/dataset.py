import numpy as np
from torch.utils.data import Dataset


class SemanticSegmentationDataset(Dataset):
    """
    A custom dataset class for semantic segmentation tasks.

    This class handles datasets containing images and their corresponding segmentation masks.
    It supports preprocessing using a feature extractor and optional Albumentations transformations.

    Attributes:
        data (Dataset): The dataset containing images and masks.
        feature_extractor (SegformerImageProcessor): The feature extractor for preprocessing images.
        transform (Compose, optional): The Albumentations transformations applied to each sample.

    Args:
        data (Dataset): The dataset containing images and masks.
        feature_extractor (SegformerImageProcessor): The feature extractor for preprocessing images.
        transform (Compose, optional): Optional Albumentations transformations to be applied.
    """

    def __init__(self, data, feature_extractor, transform=None):
        """
        Initializes the dataset with the given data, feature extractor, and optional transformations.
        """
        self.data = data
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses the image and mask at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed image and mask.
        """
        # Get image path from the loaded data
        image = self.data[idx]["image"].convert("RGB")
        # Get mask path from the loaded data
        mask = self.data[idx]["mask"].convert("L")

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            image, mask = self.transform(image=image, mask=mask)

        # Apply feature extractor
        encoded_inputs = self.feature_extractor(image, mask, return_tensors="pt")
        for k, _ in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # Remove batch dimension

        return encoded_inputs
