import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation

from ucs.utils.config import TrainingConfig
from ucs.utils.metrics import FocalLoss, SegMetrics, TestMetrics

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")  # pragma: no cover


class SegformerFinetuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the SegFormer model for semantic segmentation tasks.

    Attributes:
        model (SegformerForSemanticSegmentation): The SegFormer model for semantic segmentation.
        metrics (SegMetrics): Metrics object for tracking performance.
        criterion (FocalLoss): The loss function used for training.
        test_results (dict): Stores 'predictions' and 'ground_truths' for test evaluation.

    Hyperparameters (hparams):
        model_name (str): The SegFormer variant to use (e.g., "b0").
        max_epochs (int): Maximum number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization).
        ignore_index (int or None): Label index to ignore during training.
        weighting_strategy (str): Strategy for class weighting ('none', 'balanced', 'max', 'sum', or 'raw').
        gamma (float): Focal loss gamma parameter.
        id2label (dict): Mapping from class indices to class labels.
    """

    def __init__(self, config: TrainingConfig = None, class_weights=None, **kwargs):
        """
        Initializes the model with the given configuration, class weights, and additional hyperparameters.

        Args:
            config (TrainingConfig, optional): Training configuration containing model hyperparameters.
            class_weights (torch.Tensor, optional): Class weights for loss balancing.
            **kwargs: Additional hyperparameters to override config values.
        """

        super().__init__()
        config = config or TrainingConfig()

        # Save all hyperparameters, handle eta_min & class_weights
        self._save_hparams(config, class_weights, **kwargs)

        # Initialize the model
        self._initialize_model()

        # Initialize metrics
        self._initialize_metrics()

        # Initialize loss function
        self._initialize_loss()

        # Initialize test results storage
        self.test_results = {
            "predictions": torch.tensor([]),
            "ground_truths": torch.tensor([]),
        }

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
        self.label2id = {v: k for k, v in self.hparams.id2label.items()}
        self.num_classes = len(self.hparams.id2label)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{self.hparams.model_name}-finetuned-ade-512-512",
            num_labels=self.num_classes,
            id2label=self.hparams.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
            semantic_loss_ignore_index=self.hparams.ignore_index,
        )
        self.model.train()

    def _initialize_metrics(self):
        """
        Initializes segmentation performance metrics.
        """

        self.metrics = SegMetrics(
            self.num_classes, self.device, ignore_index=self.hparams.ignore_index
        )

    def _initialize_loss(self):
        """
        Initializes the Focal Loss function.

        Uses:
            - `gamma` hyperparameter for focusing.
            - `class_weights` if provided.
            - `ignore_index` to exclude certain labels.
        """

        self.criterion = FocalLoss(
            num_classes=self.num_classes,
            alpha=self.hparams.class_weights,
            gamma=self.hparams.gamma,
            reduction="mean",
            ignore_index=self.hparams.ignore_index,
        )

    def on_fit_start(self):
        """
        Sets the model in training mode at the start of training.
        """

        self.train()
        self.model.train()

    def on_train_start(self):
        """
        Called at the start of training, set model in training mode.
        """
        super().on_train_start()
        self.model.train()

    def on_test_start(self):
        """
        Add test-specific metrics at the start of the test phase.
        """
        super().on_test_start()
        # pylint: disable=attribute-defined-outside-init
        self.metrics = TestMetrics(
            self.num_classes, self.device, self.hparams.ignore_index
        )

    def forward(self, images, masks=None):
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Input images of shape (batch_size, num_channels, height, width).
            masks (torch.Tensor, optional): Ground truth masks of shape (batch_size, height, width). Default is None.

        Returns:
            tuple:
                loss (torch.Tensor or None): Computed loss if masks are provided, else None.
                predictions (torch.Tensor): Predicted labels of shape (batch_size, height, width), with class indices for each pixel.
        """
        outputs = self.model(pixel_values=images)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits,
            size=(images.shape[-2], images.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        loss = self.criterion(upsampled_logits, masks) if masks is not None else None
        return loss, upsampled_logits.argmax(dim=1)

    def log_step(self, stage, loss, metrics):
        """
        Logs loss and evaluation metrics.

        Args:
            stage (str): One of 'train', 'val', or 'test'.
            loss (torch.Tensor): Loss value for the current step.
            metrics (dict): Computed segmentation metrics.
        """
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        for metric_name, value in metrics.items():
            self.log(
                f"{stage}_{metric_name}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def step(self, batch, stage):
        """
        Runs a single step of training, validation, or testing.

        Args:
            batch (dict): Contains 'pixel_values' (images) and 'labels' (masks).
            stage (str): One of 'train', 'val', or 'test'.

        Returns:
            torch.Tensor: The computed loss.
        """
        images, masks = batch["pixel_values"], batch["labels"]
        loss, predicted = self(images, masks)

        self.metrics.update(predicted, masks)
        metrics = self.metrics.compute()
        self.log_step(stage, loss, metrics)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.

        Args:
            batch (dict): Contains input images and ground truth masks.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Loss, predictions, and ground truths for further aggregation.
        """
        images, masks = batch["pixel_values"], batch["labels"]
        loss, predicted = self(images, masks)

        # Compute metrics
        self.metrics.update(predicted, masks)
        metrics = self.metrics.compute()
        self.log_step("test", loss, metrics)

        # Collect predictions and ground truths for this batch
        self.test_results["predictions"] = torch.cat(
            (self.test_results["predictions"], predicted.view(-1).cpu()), dim=0
        )
        self.test_results["ground_truths"] = torch.cat(
            (self.test_results["ground_truths"], masks.view(-1).cpu()), dim=0
        )

        return loss

    def on_train_epoch_end(self):
        """
        Reset the metrics at the end of the train epoch.

        This prevents the accumulation of metric values across epochs and ensures metrics are calculated independently for each epoch.
        """
        self.metrics.reset()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        """
        Reset the metrics at the end of the validation epoch.

        This prevents the accumulation of metric values across epochs and ensures metrics are calculated independently for each epoch.
        """
        self.metrics.reset()
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        """
        Reset the metrics at the end of the test epoch.
        """
        self.metrics.reset()
        return super().on_test_epoch_end()

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

    def freeze_encoder_layers(self, blocks_to_freeze=None):
        """
        Freezes specified encoder layers to prevent weight updates.

        Args:
            blocks_to_freeze (list[str], optional): List of encoder blocks to freeze.
        """
        if blocks_to_freeze is None:
            # Freeze the first two blocks by default
            blocks_to_freeze = ["block.0"]

        for name, param in self.model.named_parameters():
            if any(block in name for block in blocks_to_freeze):
                param.requires_grad = False

    def unfreeze_encoder_layers(self, blocks_to_unfreeze=None):
        """
        Unfreezes specified encoder layers to allow training updates.

        Args:
            blocks_to_unfreeze (list[str], optional): List of encoder blocks to unfreeze.
        """

        if blocks_to_unfreeze is None:
            # Freeze the first two blocks by default
            blocks_to_unfreeze = ["block.0"]
        for name, param in self.model.named_parameters():
            if any(block in name for block in blocks_to_unfreeze):
                param.requires_grad = True

    def save_pretrained_model(self, pretrained_path):
        """
        Save the trained model in Hugging Face's Transformers-compatible format.

        Args:
            pretrained_path (str or Path): Directory where the model will be saved.

        Notes:
            This allows the saved model to be loaded later using the `from_pretrained` method.
        """

        self.model.save_pretrained(pretrained_path)  # pragma: no cover

    def calculate_confusion_matrix(self):
        """
        Calculate the confusion matrix from test predictions and ground truths.

        Returns:
            np.ndarray: The confusion matrix.
        """
        from sklearn.metrics import confusion_matrix

        predictions = self.test_results["predictions"].numpy()
        ground_truths = self.test_results["ground_truths"].numpy()

        # Reset test_results
        self.test_results = {
            "predictions": torch.tensor([]),
            "ground_truths": torch.tensor([]),
        }

        # Compute the confusion matrix
        return confusion_matrix(
            y_true=ground_truths,
            y_pred=predictions,
            labels=list(self.hparams.id2label.keys()),
        )
