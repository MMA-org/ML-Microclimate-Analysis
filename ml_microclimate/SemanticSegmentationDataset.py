from torch.utils.data import Dataset
import numpy as np


class SemanticSegmentationDataset(Dataset):
    """
    Dataset class for semantic segmentation tasks.
    Args:
        data (list): List of dictionaries with image and mask file paths.
        feature_extractor: Feature extractor for preprocessing images.
        augment (bool): If True, apply data augmentation.
    """

    def __init__(self, data, feature_extractor=None, transform=None):
        """
        Initializes the class with the provided arguments.
        """
        self.data = data  # Data loaded from Hugging Face (list of dictionaries)
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image path from the loaded data
        image = self.data[idx]['image'].convert("RGB")
        # Get mask path from the loaded data
        mask = self.data[idx]['mask'].convert("L")
        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            image, mask = self.transform(image=image, mask=mask)

        if self.feature_extractor:
            encoded_inputs = self.feature_extractor(
                image, mask, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs
