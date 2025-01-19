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
            Create callable albumination.Compose::

            transform = Augmentation()
            image,mask = transform(image,mask) # return augmented image and mask

        """
        self.transform = A.Compose([
            # Geometric Augmentations
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], p=0.5),

            # Photometric Augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            ], p=0.3),

            # Distortions
            A.OneOf([
                A.ElasticTransform(
                    alpha=1, sigma=50,
                    border_mode=0,  # Replace deprecated mode with border_mode
                    p=0.5
                ),
                A.GridDistortion(
                    num_steps=5, distort_limit=0.3,
                    border_mode=0,  # Replace deprecated mode with border_mode
                    p=0.5
                ),
                A.OpticalDistortion(
                    distort_limit=0.5, shift_limit=0.05,
                    border_mode=0,  # Replace deprecated mode with border_mode
                    p=0.5
                ),
            ], p=0.3),

            # Blur
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ], p=0.2),

            # To Tensor (must always be last)
            ToTensorV2()
        ])

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
            return augmented['image'], augmented['mask']
        else:
            augmented = self.transform(image=image)
            return augmented['image']
