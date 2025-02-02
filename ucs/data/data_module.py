from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from ucs.data.dataset import SemanticSegmentationDataset
from ucs.utils.config import DatasetConfig


class SegmentationDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for semantic segmentation tasks. This class handles dataset
    preparation, transformation, and creation of data loaders for training, validation, and testing.

    Attributes:
        dataset_path (str): Path to the dataset or dataset identifier, initialized from the config.
        batch_size (int): Batch size for the data loaders, initialized from the config.
        num_workers (int): Number of workers for data loading, initialized from the config.
        do_reduce_labels (bool): Whether to reduce label values, initialized from the config.
        pin_memory (bool): Whether to use pinned memory for faster data transfer, initialized from the config.
        transform (callable, optional): Transformations to apply to the dataset.
        persistent_workers (bool): Whether to use persistent workers in data loading.
        feature_extractor (SegformerImageProcessor): Pre-trained feature extractor initialized with the model name.
        raw_dataset (Dataset or None): The raw dataset loaded from the dataset source.
        train_dataset (Dataset or None): The processed training dataset.
        val_dataset (Dataset or None): The processed validation dataset.
        test_dataset (Dataset or None): The processed test dataset.
    """

    def __init__(self, config: DatasetConfig = None, transform=None, **kwargs):
        """
        Initializes the SegmentationDataModule with dataset configurations.

        Args:
            config (DatasetConfig, optional): Configuration object containing dataset parameters.
            transform (callable, optional): Transformations to apply to the dataset.
            **kwargs: Additional keyword arguments for overriding dataset configurations.
        """

        super().__init__()
        self._load_config(config or DatasetConfig(), kwargs)
        self.transform = transform
        self.persistent_workers = self.num_workers > 0
        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            f"nvidia/segformer-{self.model_name}-finetuned-ade-512-512",
            do_reduce_labels=self.do_reduce_labels,
        )
        self.raw_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _load_config(self, config: DatasetConfig, overrides: dict):
        """
        Loads configuration values and allows overrides from keyword arguments.

        Args:
            config (DatasetConfig): The configuration object containing dataset parameters.
            overrides (dict): A dictionary of parameter overrides.
        """
        for key, value in vars(config).items():
            setattr(self, key, overrides.get(key, value))

    def prepare_data(self):
        """
        Loads or downloads the dataset. Called once before training.
        """
        self.raw_dataset = load_dataset(self.dataset_path)

    def setup(self, stage=None):
        """
        Sets up datasets for training, validation, or testing based on the given stage.

        Args:
            stage (str, optional): The stage to set up. Options are 'fit', 'validate', or 'test'.
        """
        if self.raw_dataset is None:
            self.prepare_data()
        if stage == "fit":
            self.train_dataset = self._prepare_dataset(
                self.raw_dataset["train"], self.transform
            )
            self.val_dataset = self._prepare_dataset(self.raw_dataset["validation"])

        if stage == "test":
            self.test_dataset = self._prepare_dataset(self.raw_dataset["test"])

    def train_dataloader(self):
        """
        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """
        Returns:
            DataLoader: A PyTorch DataLoader for the validation dataset.
        """
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        """
        Returns:
            DataLoader: A PyTorch DataLoader for the test dataset.
        """
        return self._create_dataloader(self.test_dataset)

    def _prepare_dataset(self, dataset_split, transform=None):
        """
        Prepares a dataset with optional transformations.

        Args:
            dataset_split (Dataset): The dataset split to prepare (train/val/test).
            transform (callable, optional): Transformations to apply to the dataset.

        Returns:
            SemanticSegmentationDataset: The prepared dataset.
        """
        return SemanticSegmentationDataset(
            data=dataset_split,
            feature_extractor=self.feature_extractor,
            transform=transform,
        )

    def _create_dataloader(self, dataset, shuffle=False):
        """
        Creates a PyTorch DataLoader for a given dataset.

        Args:
            dataset (Dataset): The dataset to load.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

        Returns:
            DataLoader: A PyTorch DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
