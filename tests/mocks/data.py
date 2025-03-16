import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerImageProcessor


# --- Dummy load_dataset ---
def dummy_load_dataset(use_ignore_index=False):
    """
    Mimics datasets.load_dataset by returning a dictionary with splits.

    Args:
        use_ignore_index (bool): If True, set the entire first row of each mask to 255.

    Returns:
        dict: A dictionary with 'train', 'validation', and 'test' splits.
    """
    return {
        "train": DummyDatasetSplit(
            num_samples=4, num_classes=2, use_ignore_index=use_ignore_index
        ),
        "validation": DummyDatasetSplit(
            num_samples=2, num_classes=2, use_ignore_index=use_ignore_index
        ),
        "test": DummyDatasetSplit(
            num_samples=1, num_classes=2, use_ignore_index=use_ignore_index
        ),
    }


class DummyDatasetSplit:
    """
    A dummy dataset split that simulates a Hugging Face dataset split.

    Optionally, it sets the entire first row of each mask to 255 (ignore index).
    """

    def __init__(self, num_samples, num_classes=2, use_ignore_index=False):
        self.samples = []
        for i in range(num_samples):
            # Cycle through class values: 0, 1, ..., num_classes-1.
            mask_value = i % num_classes
            image = Image.new(
                "RGB", (64, 64), color=((i * 30) % 255, (i * 60) % 255, (i * 90) % 255)
            )
            mask = Image.new("L", (64, 64), color=mask_value)
            if use_ignore_index:
                mask_array = np.array(mask)
                mask_array[0, :] = 255  # Set the first row as the ignore index.
                mask = Image.fromarray(mask_array.astype("uint8"))
            self.samples.append({"image": image, "mask": mask})

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


# --- Dummy Feature Extractor ---
class DummySegformerImageProcessor:
    @classmethod
    def from_pretrained(cls, do_reduce_labels=True):
        """
        Returns an instance of the dummy processor (avoiding heavy model loading).
        """
        return cls()

    def __call__(self, image, mask, return_tensors="pt"):
        """
        Mimics preprocessing by converting a PIL image to a tensor (C, H, W)
        and converting the mask to a tensor (H, W).
        """
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_array).permute(2, 0, 1)
        mask_tensor = torch.tensor(np.array(mask).astype(int))
        return {"pixel_values": image_tensor, "labels": mask_tensor}


# --- Semantic Segmentation Dataset ---
class SemanticSegmentationDataset(Dataset):
    """
    A dataset that uses the dummy feature extractor to process images and masks.

    This dataset uses the 'validation' split from the provided data.
    """

    def __init__(self, data, transform=None):
        self.data = data["validation"]
        self.feature_extractor = DummySegformerImageProcessor.from_pretrained(
            "dummy-model"
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image, mask = sample["image"], sample["mask"]
        if self.transform:
            image, mask = self.transform(image=image, mask=mask)
        encoded_inputs = self.feature_extractor(image, mask, return_tensors="pt")
        # Remove the extra batch dimension.
        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key].squeeze(0)
        return encoded_inputs


# --- Helper Functions ---
def mock_dataset(use_ignore_index=False):
    """
    Returns an instance of SemanticSegmentationDataset using dummy data.
    """
    data = dummy_load_dataset(use_ignore_index=use_ignore_index)
    return SemanticSegmentationDataset(data=data, transform=None)


def mock_dataloader(batch_size=2, use_ignore_index=False):
    """
    Returns a DataLoader for the semantic segmentation dataset.
    """
    dataset = mock_dataset(use_ignore_index=use_ignore_index)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )


def mock_logits(labels, num_classes):
    """
    Given a ground truth mask tensor with shape [B, H, W],
    creates dummy logits with shape [B, num_classes, H, W] such that
    the argmax along the channel dimension yields the ground truth.
    """
    B, H, W = labels.shape
    logits = torch.ones((B, num_classes, H, W), dtype=torch.float)
    for b in range(B):
        for i in range(H):
            for j in range(W):
                cls = labels[b, i, j].item()
                logits[b, cls, i, j] = 10.0
    return logits


# --- Fixtures ---
@pytest.fixture
def mock_batch():
    """
    Creates a batch from the mock dataloader without ignore index.
    Returns a tuple (logits, labels), where:
      - logits is a tensor of shape [B, num_classes, H, W]
      - labels is a tensor of shape [B, H, W]
    """
    dataloader = mock_dataloader()
    data_batch = next(iter(dataloader))
    num_classes = int(torch.max(data_batch["labels"])) + 1
    logits = mock_logits(labels=data_batch["labels"], num_classes=num_classes)
    return logits, data_batch["labels"]


@pytest.fixture
def mock_batch_with_ignore():
    """
    Creates a batch from the mock dataloader with ignore index.
    Returns a tuple (logits, labels), where:
      - logits is a tensor of shape [B, num_classes, H, W]
      - labels is a tensor of shape [B, H, W]
    """
    dataloader = mock_dataloader(use_ignore_index=True)
    data_batch = next(iter(dataloader))
    num_classes = int(torch.max(data_batch["labels"])) + 1
    logits = mock_logits(labels=data_batch["labels"], num_classes=num_classes)
    return logits, data_batch["labels"]
