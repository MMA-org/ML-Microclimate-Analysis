import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from pathlib import Path
from utils.metrics import Metrics
from transformers import logging
import warnings

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')


class SegformerFinetuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Segformer model for semantic segmentation tasks.

    Args:
        id2label (dict): A dictionary mapping class IDs to class labels.
        model_name (str): The name of the Segformer model variant to use. Default is "b0".
        class_weight (torch.Tensor, optional): Class weights for the loss function. Default is None.
        metrics_interval (int): Interval at which metrics are logged. Default is 100.
        lr (float): Learning rate for the optimizer. Default is 2e-5.
        eps (float): Epsilon value for the optimizer. Default is 1e-8.

    Attributes:
        id2label (dict): A dictionary mapping class IDs to class labels.
        label2id (dict): A dictionary mapping class labels to class IDs.
        num_classes (int): The number of classes.
        model (SegformerForSemanticSegmentation): The Segformer model for semantic segmentation.
        metrics (Metrics): Metrics object for tracking performance.
        test_results (dict): Dictionary to store test predictions and ground truths.
        class_weights (torch.Tensor): Class weights for the loss function.
        criterion (nn.CrossEntropyLoss): Loss function.
    """

    def __init__(self, id2label, model_name="b0", class_weight=None, metrics_interval=100, lr=2e-5, eps=1e-8):
        super().__init__()
        self.save_hyperparameters(ignore=['id2label'])

        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.num_classes = len(id2label)

        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{model_name}-finetuned-ade-512-512",
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        self.model.train()

        # Metrics
        self.metrics = Metrics(self.num_classes, self.device)

        # Store test results
        self.test_results = {"predictions": [], "ground_truths": []}

        # Loss function (weighted cross-entropy)
        self.class_weights = class_weight
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, images, masks=None):
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Input images.
            masks (torch.Tensor, optional): Ground truth masks. Default is None.

        Returns:
            torch.Tensor: Model outputs.
        """
        outputs = self.model(pixel_values=images, labels=masks)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = self.criterion(upsampled_logits, masks)
        return loss, upsampled_logits.argmax(dim=1)

    def step(self, batch, stage):
        """
        Perform a single step in the training/validation/test loop.

        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            stage (str): The current stage (e.g., 'train', 'val', 'test').

        Returns:
            torch.Tensor: The computed loss.
        """
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self(images, masks)

        # Log loss at the step level
        self.log(f"{stage}_loss", loss, prog_bar=True)

        # Update metrics
        self.metrics.update(predicted, masks)

        # Log intermediate metrics at the step level
        step_metrics = self.metrics.compute()

        self.log(f"{stage}_mean_iou",
                 step_metrics["mean_iou"], prog_bar=True)
        self.log(f"{stage}_mean_dice",
                 step_metrics["mean_dice"], prog_bar=True)

        if stage == "test":
            # Collect test predictions and ground truths
            self.test_results["predictions"].extend(predicted.cpu().numpy())
            self.test_results["ground_truths"].extend(masks.cpu().numpy())

        return loss

    def on_train_start(self):
        """
        Called at the start of training, set model in training mode.
        """
        super().on_train_start()
        self.model.train()

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
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.step(batch, "test")

    def on_epoch_end(self, stage):
        """
        Compute and log metrics at the end of the epoch.
        """
        self.metrics.reset()

    def on_training_epoch_end(self):
        """
        Compute and log metrics at the end of the training epoch.
        """
        self.on_epoch_end("train")  # pragma: no cover

    def on_validation_epoch_end(self):
        """
        Compute and log metrics at the end of the validation epoch.
        """
        self.on_epoch_end("val")  # pragma: no cover

    def on_test_epoch_end(self):
        """
        Compute and log metrics at the end of the test epoch.
        """
        self.on_epoch_end("test")  # pragma: no cover

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            tuple: Optimizer and learning rate scheduler.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, eps=self.hparams.eps)

    def save_pretrained_model(self, pretrained_path, checkpoint_path=None):
        """
        Save the best model to a directory.

        Args:
            pretrained_path (str or Path): Directory where the model will be saved.
            checkpoint_path (str or Path, optional): Path to the checkpoint file.
        """
        if checkpoint_path and Path(checkpoint_path).exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore all warnings
                logging.set_verbosity_error()  # Suppress Transformers logging
                model = SegformerFinetuner.load_from_checkpoint(
                    checkpoint_path, id2label=self.id2label
                )
                model.model.save_pretrained(pretrained_path)
                logging.set_verbosity_warning()  # Restore Transformers logging level
        else:
            self.model.save_pretrained(pretrained_path)

    def reset_test_results(self):
        """
        Clear test predictions and ground truths.
        """
        self.test_results = {"predictions": [], "ground_truths": []}

    def get_test_results(self):
        """
        Retrieve test predictions and ground truths.
        """
        return self.test_results
