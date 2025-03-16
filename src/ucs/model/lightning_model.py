import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassConfusionMatrix
from transformers import SegformerForSemanticSegmentation

from ucs.utils import save_confusion_matrix_plot
from ucs.utils.config import TrainingConfig
from ucs.utils.metrics import CeDiceLoss, SegMetrics, TestMetrics

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")  # pragma: no cover

# TODO test metrics (compute/reset) handles by PL
# TODO docs ignore index no needed


class SegformerFinetuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the SegFormer model for semantic segmentation tasks.

    Attributes:
        model (SegformerForSemanticSegmentation): The SegFormer model for semantic segmentation.
        metrics (SegMetrics): Metrics object for tracking performance.
        criterion (CeDiceLoss): The loss function used for training.
        test_results (dict): Stores 'predictions' and 'ground_truths' for test evaluation.

    Args:
        config (TrainingConfig, optional):
            Training configuration containing model hyperparameters, stored in `self.hparams`.
        class_weights (torch.Tensor, optional):
            Class weights for loss balancing.
        **kwargs (dict): Additional hyperparameters that override `config`, such as:

            - **backbone** (str): The SegFormer variant to use (e.g., "b0").
            - **max_epochs** (int): Maximum number of training epochs.
            - **learning_rate** (float): Learning rate for the optimizer.
            - **weight_decay** (float): Weight decay (L2 regularization).
            - **weighting_strategy** (str): Strategy for class weighting ('none', 'balanced', 'max', 'sum', or 'raw').
            - **alpha** (float): CeDiceLoss loss alpha parameter.
            - **id2label** (Dict[int, str]): Mapping from class indices to class labels.
    """

    def __init__(self, config: TrainingConfig = None, class_weights=None, **kwargs):
        """
        Initializes the model with the given configuration, class weights, and additional hyperparameters.
        """

        super().__init__()
        config = config or TrainingConfig()

        # Save all hyperparameters, handle eta_min & class_weights
        self._save_hparams(config, class_weights, **kwargs)

        # Initialize the model
        self._initialize_model()

        # Initialize metrics & loss function
        self._initialize_metrics_loss()

    def _save_hparams(self, config, class_weights, **kwargs):
        """
        Saves hyperparameters and ensures eta_min is correctly set.

        Args:
            config (TrainingConfig): Configuration object containing model hyperparameters.
            class_weights (torch.Tensor or None): Class weights for loss balancing.
            **kwargs: Additional keyword arguments to override configurations.
        """

        class_weights = class_weights.tolist() if class_weights is not None else None
        self.save_hyperparameters(
            {**config.__dict__, **kwargs, "class_weights": class_weights}
        )

        if not hasattr(self.hparams, "eta_min"):
            self.hparams.eta_min = self.hparams.learning_rate * 0.01

        if self.hparams.class_weights is not None:
            self.hparams.class_weights = torch.tensor(
                self.hparams.class_weights, dtype=torch.float32
            )

    def _initialize_model(self):
        """
        Loads the SegFormer model with the correct parameters.

        Initializes:
            - The label-to-ID mapping.
            - The number of classes.
            - The model using the selected SegFormer variant.
        """
        self.labels = list(self.hparams.id2label.values())
        self.label2id = {v: k for k, v in self.hparams.id2label.items()}
        self.num_classes = len(self.hparams.id2label)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{self.hparams.backbone}-finetuned-ade-512-512",
            num_labels=self.num_classes,
            id2label=self.hparams.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.train()

    def _initialize_metrics_loss(self):
        """
        Initializes the CeDiceLoss Loss function.

        Uses:
            - `alpha` hyperparameter for focusing.
            - `class_weights` if provided.
            - `ignore_index` to exclude certain labels.
        """
        self.train_metrics = SegMetrics(self.num_classes, prefix="train")
        self.val_metrics = SegMetrics(self.num_classes, prefix="val")
        self.criterion = CeDiceLoss(
            weights=self.hparams.class_weights,
            alpha=self.hparams.alpha,
            reduction="mean",
        )

    def on_fit_start(self):
        """
        Sets the model in training mode at the start of training.
        """

        self.train()
        self.model.train()
        super().on_fit_start()

    def on_train_start(self):
        """
        Called at the start of training, set model in training mode.
        """
        self.train()
        self.model.train()
        super().on_train_start()

    def on_test_start(self):
        """
        Add test-specific metrics at the start of the test phase.
        """
        super().on_test_start()
        # pylint: disable=attribute-defined-outside-init
        self.confmat = MulticlassConfusionMatrix(
            num_classes=self.num_classes,
            ignore_index=255,
            normalize="true",
        )
        self.test_metrics = TestMetrics(
            self.num_classes,
            prefix="test",
        )

    def forward(self, pixel_values, labels=None):
        """
        Forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input pixel_values of shape (batch_size, num_channels, height, width).
            labels (torch.Tensor, optional): Ground truth labels of shape (batch_size, height, width). Default is None.

        Returns:
            tuple:
                loss (torch.Tensor or None): Computed loss if labels are provided, else None.
                predictions (torch.Tensor): Predicted labels of shape (batch_size, height, width), with class indices for each pixel.
        """
        outputs = self.model(pixel_values=pixel_values)
        logits = nn.functional.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits.argmax(dim=1)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss, predicted = self(batch["pixel_values"], batch["labels"])

        self.train_metrics(predicted, batch["labels"])

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.train_metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss, predicted = self(batch["pixel_values"], batch["labels"])
        self.val_metrics(predicted, batch["labels"])

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.val_metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.

        Args:
            batch (dict): Contains input images and ground truth masks.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss, predicted = self(batch["pixel_values"], batch["labels"])

        metrics = self.test_metrics(predicted, batch["labels"])
        self.confmat.update(predicted.flatten(), batch["labels"].flatten())
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Set up the optimizer and learning rate scheduler.

        Returns:
            tuple: A list containing:
                - optimizer (torch.optim.AdamW): Optimizer configured with weight decay and learning rate.
                - scheduler (dict): CosineAnnealingLR scheduler, which reduces the learning rate following a cosine schedule.
        """

        # Set up the optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.eta_min
            ),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def save_pretrained_model(self, pretrained_path):
        """
        Save the trained model in Hugging Face's Transformers-compatible format.

        Args:
            pretrained_path (str or Path): Directory where the model will be saved.

        Notes:
            This allows the saved model to be loaded later using the `from_pretrained` method.
        """

        self.model.save_pretrained(pretrained_path)  # pragma: no cover

    def calculate_confusion_matrix(self, save_path):
        """
        Calculate the confusion matrix from test predictions and ground truths.

        Returns:
            np.ndarray: The confusion matrix.
        """
        save_confusion_matrix_plot(
            self.confmat,
            save_path=save_path,
            metrics=self.test_metrics,
            labels=self.labels,
        )
        self.confmat.reset()
