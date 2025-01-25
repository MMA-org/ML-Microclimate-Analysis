import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from utils.metrics import SegMetrics, CeDiceLoss, TestMetrics
from torch.optim.lr_scheduler import CosineAnnealingLR

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')  # pragma: no cover


class SegformerFinetuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Segformer model for semantic segmentation tasks.

    Args:
        id2label (dict): A dictionary mapping class IDs to class labels. Ensure all class IDs in the dataset are included.
        model_name (str): The name of the Segformer model variant to use. Default is "b0".
        max_epochs (int): Maximum number of training epochs. Default is 50.
        lr (float): Learning rate for the optimizer. Default is 2e-5.
        alpha (float): Weighting factor for Dice loss. Default is 0.7.
        beta (float): Weighting factor for cross-entropy loss. Default is 0.3.
        class_weight (torch.Tensor, optional): Class weights for the loss function. Default is None.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default is None.
        weight_decay (float): Weight decay (L2 regularization) coefficient for the optimizer. Helps prevent overfitting. Default is 1e-3.

    Attributes:
        id2label (dict): A dictionary mapping class IDs to class labels.
        label2id (dict): A dictionary mapping class labels to class IDs.
        num_classes (int): The number of classes.
        model (SegformerForSemanticSegmentation): The Segformer model for semantic segmentation.
        metrics (SegMetrics): Metrics object for tracking performance.
        class_weights (torch.Tensor): Class weights for the loss function.
        lr (float): Learning rate for the optimizer.
        criterion (CeDiceLoss): Combined cross-entropy and Dice loss function for training.
        test_results (dict): Contains 'predictions' and 'ground_truths' tensors for calculating confusion matrix after tests.
    """

    def __init__(self, id2label, model_name="b0", max_epochs=50, lr=2e-5, alpha=0.7, beta=0.3, class_weight=None, ignore_index=None, weight_decay=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['id2label'])

        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.num_classes = len(id2label)
        self.learning_rate = lr
        self.ignore_index = ignore_index
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{model_name}-finetuned-ade-512-512",
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        self.model.train()
        # Store test results
        self.test_results = {"predictions": torch.tensor(
            []), "ground_truths": torch.tensor([])}  # Clear previous test results
        # Initialize metrics and move to the correct device
        self.metrics = SegMetrics(
            self.num_classes, self.device, ignore_index=self.ignore_index)

        # Initialize the loss function (CeDiceLoss)
        self.criterion = CeDiceLoss(
            num_classes=self.num_classes,
            weights=class_weight,
            alpha=alpha,
            beta=beta,
            reduction='mean',
            ignore_index=self.ignore_index
        )

    def on_fit_start(self):
        """
        set model in training mode.
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
        self.metrics = TestMetrics(
            self.num_classes, self.device, self.ignore_index)

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
            outputs.logits, size=(images.shape[-2], images.shape[-1]),
            mode="bilinear", align_corners=False
        )
        loss = self.criterion(
            upsampled_logits, masks) if masks is not None else None
        return loss, upsampled_logits.argmax(dim=1)

    def log_step(self, stage, loss, metrics):
        # Logging loss and metrics
        self.log(f"{stage}_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)

        for metric_name, value in metrics.items():
            self.log(f"{stage}_{metric_name}", value,
                     prog_bar=True, on_step=False, on_epoch=True)

    def step(self, batch, stage):
        """
        Perform a single step in the training, validation loop.

        Args:
            batch (dict): A batch of data containing:
                - 'pixel_values' (torch.Tensor): Input images of shape (batch_size, num_channels, height, width).
                - 'labels' (torch.Tensor): Ground truth masks of shape (batch_size, height, width).
            stage (str): The current stage, one of 'train', 'val', or 'test'.

        Returns:
            torch.Tensor: The computed loss for the current step.
        """
        images, masks = batch['pixel_values'], batch['labels']
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
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self(images, masks)

        # Compute metrics
        self.metrics.update(predicted, masks)
        metrics = self.metrics.compute()
        self.log_step("test", loss, metrics)

        # Collect predictions and ground truths for this batch
        self.test_results["predictions"] = torch.cat(
            (self.test_results["predictions"], predicted.view(-1).cpu()), dim=0)
        self.test_results["ground_truths"] = torch.cat(
            (self.test_results["ground_truths"], masks.view(-1).cpu()), dim=0)

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
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.learning_rate*0.01),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def freeze_encoder_layers(self, blocks_to_freeze=None):
        """
        Freeze the specified encoder layers to prevent updates during training.

        Args:
            blocks_to_freeze (list[str], optional): List of encoder blocks to freeze. Default is ["block.0"].

        Notes:
            Use this method to selectively freeze layers during fine-tuning to preserve pre-trained weights for certain layers.
        """
        if blocks_to_freeze is None:
            # Freeze the first two blocks by default
            blocks_to_freeze = ["block.0"]

        for name, param in self.model.named_parameters():
            if any(block in name for block in blocks_to_freeze):
                param.requires_grad = False

    def unfreeze_encoder_layers(self, blocks_to_unfreeze=None):
        """
        Unfreeze the specified encoder layers to allow updates during training.

        Args:
            blocks_to_unfreeze (list[str], optional): List of encoder blocks to unfreeze. Default is ["block.0"].

        Notes:
            Use this method to selectively unfreeze layers when transitioning to full model fine-tuning.
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
        self.test_results = {"predictions": torch.tensor(
            []), "ground_truths": torch.tensor([])}

        # Compute the confusion matrix
        return confusion_matrix(
            y_true=ground_truths, y_pred=predictions, labels=list(
                self.id2label.keys())
        )
