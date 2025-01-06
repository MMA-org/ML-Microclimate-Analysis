from torch.utils.data import Dataset
import numpy as np


class SemanticSegmentationDataset(Dataset):
    """
    A custom dataset class for semantic segmentation tasks.

    This class is designed to handle datasets containing images and their corresponding segmentation masks.
    It supports preprocessing using a feature extractor and optional Albumentations transformations.

    Args:
        data (Dataset): 
            The dataset containing images and masks. Each sample in the dataset should be a dictionary with 
            keys for the image and the mask.
        feature_extractor (SegformerImageProcessor): 
            The feature extractor for preprocessing images. Typically used for resizing, normalization, and 
            converting images into tensors.
        transform (Compose, optional): 
            Optional Albumentations transformations to be applied on the images and masks during training or evaluation.

    Attributes:
        data (Dataset): 
            The dataset containing images and masks.
        feature_extractor (SegformerImageProcessor): 
            The feature extractor for preprocessing images.
        transform (Compose, optional): 
            The Albumentations transformations applied on each sample.
    """

    def __init__(self, data, feature_extractor, transform=None):
        self.data = data
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed image and mask.
        """
        # Get image path from the loaded data
        image = self.data[idx]['image'].convert("RGB")
        # Get mask path from the loaded data
        mask = self.data[idx]['mask'].convert("L")

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Apply feature extractor
        encoded_inputs = self.feature_extractor(
            image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # Remove batch dimension

        return encoded_inputs
