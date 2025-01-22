from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from .dataset import SemanticSegmentationDataset
from datasets import load_dataset


class Loader:
    """
    A utility class for loading datasets and creating PyTorch DataLoaders for semantic segmentation tasks.

    Args:
        dataset_path (str): Path to the dataset.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        model_name (str): Name of the model used for feature extraction.

    Attributes:
        dataset_path (str): Path to the dataset.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        model_name (str): Name of the model used for feature extraction.
        dataset (Dataset): A dataset object loaded from the specified path.
        feature_extractor (SegformerImageProcessor): Preprocessor for images and masks.
    """

    def __init__(self, dataset_path, batch_size, num_workers, model_name):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load dataset
        self.dataset = load_dataset(self.dataset_path)
        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            f"nvidia/segformer-{model_name}-finetuned-ade-512-512",
            do_reduce_labels=False
        )

    def get_dataloader(self, split: str, shuffle: bool = False, transform=None) -> DataLoader:
        """
        Creates a PyTorch DataLoader for the specified dataset split.

        Args:
            split (str): The dataset split to load (e.g., "train", "validation", "test").
            shuffle (bool): Whether to shuffle the data. Defaults to False.
            transform (Compose, optional): 
                Optional Albumentations transformations to be applied on the images and masks during training or evaluation.

        Returns:
            DataLoader: A PyTorch DataLoader for the specified dataset split.

        Example:
            Create a data loader for the training dataset::

                # no augmentation
                loader = Loader(dataset_path, batch_size, num_workers, model_name)
                train_loader = loader.get_dataloader("train", shuffle=True)

                from data.transform import Augmentation
                # with custome augmentation
                loader = Loader(dataset_path, batch_size, num_workers, model_name) # augmentation is callable albumentations.Compose
                train_loader_with_augmentation = loader.get_dataloader("train",shuffle=True,transform=Augmentation())

        """
        transform = transform
        dataset = SemanticSegmentationDataset(
            data=self.dataset[split],
            feature_extractor=self.feature_extractor,
            transform=transform
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            pin_memory=True
        )
