import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from torchmetrics import JaccardIndex, Dice
from pathlib import Path


class SegformerFinetuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Segformer model for semantic segmentation tasks.
    """

    def __init__(self, id2label, model_name="b0", metrics_interval=100, lr=2e-5, eps=1e-8, step_size=10, gamma=0.1):
        super(SegformerFinetuner, self).__init__()
        self.metrics_interval = metrics_interval
        self.id2label = id2label
        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{model_name}-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        # Metrics
        self.iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes)
        self.dice = Dice(average='micro', num_classes=self.num_classes)

        # Store outputs for validation and testing
        self.val_outputs = []
        self.test_outputs = []
        self.test_predictions = []
        self.test_ground_truths = []  # Added

    def forward(self, images, masks=None):
        """
        Forward pass through the Segformer model.
        """
        outputs = self.model(pixel_values=images, labels=masks)
        return outputs

    def forward_pass(self, images, masks):
        """
        Perform a forward pass and process logits into predictions.
        """
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]

        # Resize logits to match mask dimensions
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)

        return loss, predicted

    def process_metrics(self, predicted, masks):
        """
        Process predictions and masks to ensure they are compatible with metrics.
        """
        predicted = predicted.view(-1)
        masks = masks.view(-1)

        # Mask out ignored pixels (e.g., 255)
        masks[masks == 255] = 5
        predicted = predicted[masks]
        masks = masks[masks]

        return predicted, masks

    def training_step(self, batch, batch_idx):
        """
        Execute one training step.
        """
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)

        # Process metrics
        predicted, masks = self.process_metrics(predicted, masks)

        # Update metrics
        self.iou(predicted, masks)
        self.dice(predicted, masks)

        # Log training loss
        self.log("loss", loss, prog_bar=True)

        # Log metrics at intervals
        if batch_idx % self.metrics_interval == 0:
            iou = self.iou.compute()
            dice = self.dice.compute()
            self.log("iou", iou, prog_bar=True)
            self.log("dice", dice, prog_bar=True)

            # Reset metrics after logging
            self.iou.reset()
            self.dice.reset()

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Execute one validation step.
        """
        images, masks = batch['pixel_values'], batch['labels']
        print(torch.unique(masks))
        loss, predicted = self.forward_pass(images, masks)

        # Process metrics
        predicted, masks = self.process_metrics(predicted, masks)

        # Update metrics
        self.iou(predicted, masks)
        self.dice(predicted, masks)

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True)
        self.val_outputs.append({'val_loss': loss})
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        """
        Compute and log validation metrics at the end of an epoch.
        """
        iou = self.iou.compute()
        dice = self.dice.compute()
        avg_val_loss = torch.stack([x["val_loss"]
                                   for x in self.val_outputs]).mean()

        metrics = {"val_loss": avg_val_loss,
                   "val_mean_iou": iou, "val_mean_dice": dice}
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)

        # Reset metrics after logging
        self.iou.reset()
        self.dice.reset()
        self.val_outputs.clear()
        return metrics

    def test_step(self, batch, batch_idx):
        """
        Execute one test step.
        """
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)

        # Process metrics
        predicted, masks = self.process_metrics(predicted, masks)

        # Update metrics
        self.iou(predicted, masks)
        self.dice(predicted, masks)

        # Collect predictions and ground truths
        self.test_predictions.extend(predicted.cpu().numpy())
        self.test_ground_truths.extend(masks.cpu().numpy())

        # Log test loss
        self.log("test_loss", loss, prog_bar=True)
        self.test_outputs.append({'test_loss': loss})
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        """
        Compute and log test metrics at the end of the test epoch.
        """
        iou = self.iou.compute()
        dice = self.dice.compute()
        avg_test_loss = torch.stack([x["test_loss"]
                                    for x in self.test_outputs]).mean()

        metrics = {"test_loss": avg_test_loss,
                   "test_mean_iou": iou, "test_mean_dice": dice}
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)

        # Reset metrics after logging
        self.iou.reset()
        self.dice.reset()
        self.test_outputs.clear()
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def save_pretrained_model(self, pretrained_path, checkpoint_callback=None):
        """
        Save the best model to a versioned directory.

        Args:
            pretrained_path (str or Path): Directory where the model will be saved.
            checkpoint_callback (ModelCheckpoint, optional): The ModelCheckpoint callback used during training.
        """
        # If a checkpoint callback is provided, use it to find the best model checkpoint
        if checkpoint_callback and checkpoint_callback.best_model_path:
            # Load the best model weights
            best_model_path = checkpoint_callback.best_model_path
            # Load the best model weights
            checkpoint = torch.load("path_to_checkpoint.ckpt")
            state_dict = checkpoint["state_dict"]
            clean_state_dict = {k.replace("model.", "").replace(
                "module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(
                torch.load(best_model_path), strict=False)

            # Save the model to the specified directory
            self.model.save_pretrained(pretrained_path)
            print(f"Best model saved to {pretrained_path}")
        else:
            # If no checkpoint callback or no best model path, just save the model
            self.model.save_pretrained(pretrained_path)
            print(f"Model saved to {pretrained_path}")
