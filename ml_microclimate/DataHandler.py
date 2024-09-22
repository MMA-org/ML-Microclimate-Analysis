# filename: dataset_handler.py

import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
from ml_microclimate.SemanticSegmentationDataset import SemanticSegmentationDataset


class Transform:
    def __init__(self):
        # Initialize the transformation pipeline using Albumentations
        self.transform = A.Compose([
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

    def __call__(self, image, mask):
        # Apply the transformation to the image and mask
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']


class DatasetHandler:
    def __init__(self, dataset_name, model_name="nvidia/segformer-b0-finetuned-ade-512-512", transform=None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.transform = Transform() if transform != None else None

        # Load the dataset
        self.dataset = load_dataset(self.dataset_name)
        self.train_data = self.dataset['train']
        self.validation_data = self.dataset['validation']
        self.test_data = self.dataset['test']

        # Initialize the feature extractor
        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            self.model_name)
        self.feature_extractor.do_reduce_labels = False

    def get_datasets(self):
        # Create datasets
        train_dataset = SemanticSegmentationDataset(
            data=self.train_data,
            feature_extractor=self.feature_extractor,
            transform=self.transform
        )
        validation_dataset = SemanticSegmentationDataset(
            data=self.validation_data,
            feature_extractor=self.feature_extractor
        )
        test_dataset = SemanticSegmentationDataset(
            data=self.test_data,
            feature_extractor=self.feature_extractor
        )
        return train_dataset, validation_dataset, test_dataset

    def get_dataloaders(self, batch_size=8):
        train_dataset, validation_dataset, test_dataset = self.get_datasets()
        num_workers = os.cpu_count()
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validation_dataloader = DataLoader(
            validation_dataset, batch_size=batch_size, num_workers=num_workers)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers)
        return train_dataloader, validation_dataloader, test_dataloader
