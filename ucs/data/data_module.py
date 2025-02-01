from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from datasets import load_dataset
from ucs.data.dataset import SemanticSegmentationDataset
from ucs.utils.config import DatasetConfig


class SegmentationDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for semantic segmentation tasks. This class handles dataset
    preparation, transformation, and creation of data loaders for training, validation, and testing.

    Attributes:
        dataset_path (str): Path to the dataset or dataset identifier.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of workers for data loading.
        model_name (str): The model name used to load the feature extractor.
        id2label (dict): A dictionary mapping class IDs to class labels.
        transform (callable, optional): Transformations to apply to the dataset.
        class_weights_path (str, optional): Path to save or load the class weights.
        weighting_strategy (str,optional): Normalization method for class weights ('none', 'balanced', etc.).
        ignore_index (int, optional): Index to ignore during class weight computation.
        class_weights (torch.Tensor or None): Tensor containing class weights.
        feature_extractor (SegformerImageProcessor): Pre-trained feature extractor.

    Methods:
        prepare_data(): Prepares the raw dataset. Called once during setup.
        setup(stage): Sets up datasets for the specified stage (train/val/test).
        train_dataloader(): Returns a DataLoader for the training dataset.
        val_dataloader(): Returns a DataLoader for the validation dataset.
        test_dataloader(): Returns a DataLoader for the test dataset.
    """

    def __init__(self, config: DatasetConfig = None, transform=None, **kwargs):
        """
        Initializes the SegmentationDataModule.

        Args:
            dataset_path (str): Path to the dataset or dataset identifier.
            batch_size (int): Batch size for the data loaders.
            num_workers (int): Number of workers for data loading.
            model_name (str): The model name used to load the feature extractor.
            id2label (dict): A dictionary mapping class IDs to class labels.
            transform (callable, optional): Transformations to apply to the dataset.
            class_weights_path (str, optional): Path to save or load the class weights.
            weighting_strategy (str,optional): Normalization method for class weights ('raw', 'balanced', etc.).
            ignore_index (int, optional): Index to ignore during class weight computation.
        """
        super().__init__()
        self._load_config(config or DatasetConfig(), kwargs)
        self.transform = transform
        self.persistent_workers = True if self.num_workers > 0 else False
        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            f"nvidia/segformer-{self.model_name}-finetuned-ade-512-512", do_reduce_labels=self.do_reduce_labels
        )

    def _load_config(self, config: DatasetConfig, overrides: dict):
        """Assign config values to self, allowing overrides from kwargs."""
        for key, value in vars(config).items():
            setattr(self, key, overrides.get(key, value))

    def prepare_data(self):
        """
        Prepares the raw dataset. This method is called only once, typically to load
        or download the dataset.
        """
        self.raw_dataset = load_dataset(self.dataset_path)

    def setup(self, stage=None):
        """
        Sets up the datasets for the specified stage (train, validation, or test).

        Args:
            stage (str, optional): The stage to set up. Options are 'fit', 'validate', 'test'.
        """

        self.train_dataset = self._prepare_dataset(
            self.raw_dataset["train"], self.transform)
        self.val_dataset = self._prepare_dataset(
            self.raw_dataset["validation"])

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
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers, pin_memory=self.pin_memory
        )
