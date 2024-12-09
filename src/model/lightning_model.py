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

    def __init__(self, id2label, model_name="b0", metrics_interval=100, lr=2e-5, eps=1e-8):
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
        self.iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes)
        self.dice = Dice(average='micro', num_classes=self.num_classes)

        # Store test results
        self.test_results = {"predictions": [], "ground_truths": []}

    def forward(self, images, masks=None):
        return self.model(pixel_values=images, labels=masks)

    def compute_loss_and_predictions(self, images, masks):
        outputs = self(images, masks)
        upsampled_logits = nn.functional.interpolate(
            outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        return outputs.loss, upsampled_logits.argmax(dim=1)

    def update_metrics_and_log(self, predicted, masks, stage):
        predicted, masks = predicted.view(-1), masks.view(-1)
        self.iou(predicted, masks)
        self.dice(predicted, masks)

        # Log metrics after loss
        self.log(f"{stage}_mean_iou", self.iou.compute(), prog_bar=True)
        self.log(f"{stage}_mean_dice", self.dice.compute(), prog_bar=True)

    def step(self, batch, stage):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.compute_loss_and_predictions(images, masks)

        # Log loss before metrics
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.update_metrics_and_log(predicted, masks, stage)
        return loss

    def on_train_start(self):
        super().on_train_start()
        self.model.train()

    def training_step(self, batch, batch_idx):
        assert self.training, "LightningModule is not in training mode."
        assert self.model.training, "Model is not in training mode."
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        assert not self.training, "LightningModule is not in eval mode."
        assert not self.model.training, "Model is not in eval mode."
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.compute_loss_and_predictions(images, masks)

        # Log loss before metrics
        self.log("test_loss", loss, prog_bar=True)
        self.update_metrics_and_log(predicted, masks, "test")

        # Collect test predictions and ground truths
        self.test_results["predictions"].extend(predicted.cpu().numpy())
        self.test_results["ground_truths"].extend(masks.cpu().numpy())

        return loss

    def on_epoch_end(self, stage):
        self.iou.reset()
        self.dice.reset()

    def on_training_epoch_end(self):
        self.on_epoch_end("train")

    def on_validation_epoch_end(self):
        self.on_epoch_end("val")

    def on_test_epoch_end(self):
        self.on_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, eps=self.hparams.eps)

    def save_pretrained_model(self, pretrained_path, checkpoint_path=None):
        """
        Save the best model to a directory.

        Args:
            pretrained_path (str or Path): Directory where the model will be saved.
            checkpoint_path (str or Path, optional): Path to the checkpoint file.
        """
        if checkpoint_path and Path(checkpoint_path).exists():
            model = SegformerFinetuner.load_from_checkpoint(
                checkpoint_path, id2label=self.id2label
            )
            model.model.save_pretrained(pretrained_path)
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
