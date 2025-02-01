import albumentations as A
from albumentations.pytorch import ToTensorV2


class Augmentation:
    """
    A class to define and manage data augmentation pipelines for training.

    Attributes:
        transform (albumentations.Compose):
            A composition of augmentation transformations applied to training images and masks.
    """

    def __init__(self):
        """
        Initializes the Augmentation class with a set of training transformations.

        The transformations include geometric augmentations (e.g., flips, rotations),
        photometric adjustments (e.g., brightness, contrast), distortions, occlusion handling,
        and tensor conversion.
        Example:
            Create callable albumentations.Compose::

            transform = Augmentation()
            # return augmented image and mask
            image, mask = transform(image, mask)
        """
        self.transform = A.Compose(
            [
                # Geometric Augmentations
                # Only horizontal flips, no 90-degree rotations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),  # Added vertical flip
                A.Affine(
                    translate_percent=0.0625, scale=(0.9, 1.1), rotate=20, p=0.5
                ),  # Replaces ShiftScaleRotate
                # Helps small objects
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                # Photometric Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.2
                ),
                A.GaussNoise(p=0.3),  # Added Gaussian noise
                A.CLAHE(clip_limit=2.0, p=0.2),  # Enhances fine details
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=3, p=0.2),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                    ],
                    p=0.4,
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, mask=None):
        """
        Apply the training transforms to the given image and mask.

        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray, optional): The corresponding mask. Default is None.

        Returns:
            tuple (Tensor, Tensor): Transformed image and mask (if provided), or only the transformed image.
        """
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"]
        augmented = self.transform(image=image)
        return augmented["image"]
