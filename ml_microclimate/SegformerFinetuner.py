import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from torchmetrics import Dice, JaccardIndex


class SegformerFinetuner(pl.LightningModule):
    """
    A PyTorch Lightning Module for fine-tuning the SegFormer model.
    Args:
        id2label (dict): Mapping of class IDs to labels.
        train_dataloader (DataLoader): Dataloader for training data.
        validation_dataloader (DataLoader): Dataloader for validation data.
        test_dataloader (DataLoader): Dataloader for test data.
        metrics_interval (int): Interval for logging metrics during training.
    """

    def __init__(self, id2label, train_dataloader=None, validation_dataloader=None, test_dataloader=None, metrics_interval=100):
        """
        Initializes the class with the provided arguments.
        """
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = validation_dataloader
        self.test_dl = test_dataloader

        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.ignore_index = 0  # Set the ignore index to 0

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        self.train_mean_iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes, ignore_index=0)
        self.train_dice = Dice(
            average='micro', num_classes=self.num_classes, ignore_index=0)

        self.val_mean_iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes, ignore_index=0)
        self.val_dice = Dice(
            average='micro', num_classes=self.num_classes, ignore_index=0)

        self.test_mean_iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes, ignore_index=0)
        self.test_dice = Dice(
            average='micro', num_classes=self.num_classes, ignore_index=0)

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return (outputs)

    def forward_pass(self, images, masks):
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)

        return loss, predicted

    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)

        # Update torchmetrics
        self.train_mean_iou(predicted, masks)
        self.train_dice(predicted, masks)

        if batch_nb % self.metrics_interval == 0:
            iou = self.train_mean_iou.compute()
            dice = self.train_dice.compute()

            # Log metrics
            self.log('train_mean_iou', iou, on_step=True,
                     on_epoch=True, prog_bar=True)
            self.log('train_dice', dice, on_step=True,
                     on_epoch=True, prog_bar=True)

            # Reset metrics after logging
            self.train_mean_iou.reset()
            self.train_dice.reset()

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)

        # Update torchmetrics
        self.val_mean_iou(predicted, masks)
        self.val_dice(predicted, masks)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        iou = self.val_mean_iou.compute()
        dice = self.val_dice.compute()
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        # Log metrics
        self.log('val_mean_iou', iou, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_epoch=True, prog_bar=True)
        self.log('val_loss', avg_val_loss, on_epoch=True, prog_bar=True)

        # Reset metrics after logging
        self.val_mean_iou.reset()
        self.val_dice.reset()

    def test_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)

        # Add batch to combined metrics
        self.test_mean_iou(predicted, masks)
        self.test_dice(predicted, masks)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        iou = self.test_mean_iou.compute()
        dice = self.test_dice.compute()
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        # Log test metrics
        self.log('test_mean_iou', iou, on_epoch=True, prog_bar=True)
        self.log('test_dice', dice, on_epoch=True, prog_bar=True)
        self.log('test_loss', avg_test_loss, on_epoch=True, prog_bar=True)

        # Reset metrics after logging
        self.test_mean_iou.reset()
        self.test_dice.reset()

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
