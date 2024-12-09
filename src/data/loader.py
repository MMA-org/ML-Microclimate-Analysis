# data/loader.py
from torch.utils.data import DataLoader
import cv2
from transformers import SegformerImageProcessor
from .dataset import SemanticSegmentationDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
from datasets import load_dataset


class Loader:
    """
    A module for loading datasets and creating data loaders for semantic segmentation tasks.

    Args:
        config (dict): Configuration dictionary containing training parameters.

    Attributes:
        dataset (Dataset): The loaded dataset.
        batch_size (int): The batch size for the data loader.
        num_workers (int): The number of worker processes for data loading.
        feature_extractor (SegformerImageProcessor): The feature extractor for preprocessing images.
    """

    def __init__(self, config):
        # Initialize paths and settings from the config
        self.dataset_path = config.dataset_path
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers

        # Load dataset
        self.dataset = load_dataset(self.dataset_path)

        # Initialize feature extractor
        model_name = config.training.model_name
        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            f"nvidia/segformer-{model_name}-finetuned-ade-512-512",
            do_reduce_labels=False
        )

    def get_transforms(self):
        """
        Returns the data augmentation transforms to be applied to the training data.

        Returns:
            A.Compose: A composition of data augmentation transforms.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.3),
            ToTensorV2()
        ])

    def get_dataloader(self, split, shuffle=False):
        """
        Creates a data loader for the specified dataset split.

        Args:
            split (str): The dataset split to load (e.g., 'train', 'val', 'test').
            shuffle (bool): Whether to shuffle the data. Default is False.

        Returns:
            DataLoader: A PyTorch DataLoader for the specified dataset split.
        """
        dataset = SemanticSegmentationDataset(
            data=self.dataset[split],
            feature_extractor=self.feature_extractor,
            transform=self.get_transforms() if split == "train" else None,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))
