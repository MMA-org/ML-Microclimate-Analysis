from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from .dataset import SemanticSegmentationDataset
from datasets import load_dataset
from utils import load_class_weights, save_class_weights
from utils.metrics import compute_class_weights
from pathlib import Path


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

    def __init__(self, dataset_path, batch_size, num_workers, model_name, id2label, transform=None, class_weights_path=None, weighting_strategy='raw', ignore_index=None):
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
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.weighting_strategy = weighting_strategy
        self.class_weights_path = class_weights_path
        self.id2label = id2label
        self.ignore_index = ignore_index
        self.class_weights = None
        self.pin_memory = True
        self.persistent_workers = True if num_workers > 0 else False
        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            f"nvidia/segformer-{model_name}-finetuned-ade-512-512", do_reduce_labels=False
        )

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
        if stage == "fit":
            self.train_dataset = self._prepare_dataset(
                self.raw_dataset["train"], self.transform)
            self.val_dataset = self._prepare_dataset(
                self.raw_dataset["validation"])
            self.class_weights = self._compute_class_weights()

        elif stage == "test":
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

    def _compute_class_weights(self):
        """
        Computes class weights based on the selected weighting strategy.

        Returns:
            torch.Tensor or None: A tensor containing class weights, or None if no weighting is applied.
        """
        if self.weighting_strategy == "none":
            return None

        weights_file = Path(self.class_weights_path)
        if weights_file.exists():
            return load_class_weights(weights_file)
        # Compute class weights based on the weighting strategy
        class_weights = compute_class_weights(
            dataloader=self.train_dataloader(),
            num_classes=len(self.id2label),
            normalize=self.weighting_strategy,  # Pass the strategy here
            ignore_index=self.ignore_index,
        )
        save_class_weights(weights_file, class_weights)
        return class_weights

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
