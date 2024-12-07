from torch.utils.data import Dataset
import numpy as np


class SemanticSegmentationDataset(Dataset):
    """
    A custom dataset class for semantic segmentation tasks.

    Args:
        data (Dataset): The dataset containing images and masks.
        feature_extractor (SegformerImageProcessor): The feature extractor for preprocessing images.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        data (Dataset): The dataset containing images and masks.
        feature_extractor (SegformerImageProcessor): The feature extractor for preprocessing images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data, feature_extractor, transform=None):
        # Data loaded from Hugging Face (list of dictionaries)
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
            image, mask = self.transform(image=image, mask=mask)

        # Apply feature extractor
        encoded_inputs = self.feature_extractor(
            image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # Remove batch dimension

        return encoded_inputs
