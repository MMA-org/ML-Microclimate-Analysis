from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from .dataset import SemanticSegmentationDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
from datasets import load_dataset


class Loader:
    """
    A utility class for loading datasets and creating PyTorch DataLoaders for semantic segmentation tasks.

    Args:
        config (Config): Configuration dictionary containing parameters for dataset loading and training.

    Attributes:
        dataset_path (str): path to the dataset as specified in the configuration.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        feature_extractor (SegformerImageProcessor): Preprocessor for images and masks.
    """

    def __init__(self, config):
        self.dataset_path = config.dataset.dataset_path
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
        Returns the data augmentation transforms for training data.

        The transformations include random horizontal and vertical flips, rotations,
        brightness/contrast adjustments, and other augmentations.

        Returns:
            Compose: A composition of augmentations applied to training images.
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

    def get_dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        """
        Creates a PyTorch DataLoader for the specified dataset split.

        Args:
            split (str): The dataset split to load (e.g., "train", "validation", "test").
            shuffle (bool): Whether to shuffle the data. Defaults to False.

        Returns:
            DataLoader: A PyTorch DataLoader for the specified dataset split.

        Example:
            Create a data loader for the training dataset::

                loader = Loader(config)
                train_loader = loader.get_dataloader("train", shuffle=True)
        """
        dataset = SemanticSegmentationDataset(
            data=self.dataset[split],
            feature_extractor=self.feature_extractor,
            transform=self.get_transforms() if split == "train" else None,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=True
        )
