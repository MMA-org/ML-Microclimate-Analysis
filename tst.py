import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import torch
from data.loader import Loader
from utils import Config


def get_transform_pipeline():
    """
    Returns the data augmentation and transformation pipeline.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(interpolation=cv2.INTER_NEAREST),
                    A.GridDistortion(interpolation=cv2.INTER_NEAREST),
                    A.OpticalDistortion(interpolation=cv2.INTER_NEAREST),
                ],
                p=0.3,
            ),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def debug_mask_transformation():
    """
    Debugs mask transformation pipeline to ensure values remain consistent.
    """
    transform_pipeline = get_transform_pipeline()

    # Example mask
    sample_mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    print("Original mask unique values:", np.unique(sample_mask))

    augmented = transform_pipeline(
        image=np.zeros_like(sample_mask), mask=sample_mask)
    transformed_mask = augmented["mask"]
    print("Transformed mask unique values:", np.unique(transformed_mask))

    # Convert to tensor and check unique values
    tensor_mask = torch.tensor(transformed_mask, dtype=torch.long)
    print("Tensor mask unique values:", torch.unique(tensor_mask))


def debug_dataloader():
    """
    Loads and debugs the validation DataLoader.
    """
    config = Config()
    loader = Loader(config)

    val_loader = loader.get_dataloader("validation", shuffle=False)
    for batch in val_loader:
        print("Batch labels unique values:", torch.unique(batch["labels"]))
        break


if __name__ == "__main__":
    # Debug transformations
    debug_mask_transformation()

    # Debug dataloader
    debug_dataloader()
