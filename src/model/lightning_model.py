import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
from torchmetrics import JaccardIndex, Dice


class SegformerFinetuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Segformer model for semantic segmentation tasks.

    Args:
        id2label (dict): A dictionary mapping class IDs to class labels.
        metrics_interval (int): Interval at which metrics are logged. Default is 100.
        lr (float): Learning rate for the optimizer. Default is 2e-5.
        eps (float): Epsilon value for the optimizer. Default is 1e-8.
        step_size (int): Step size for the learning rate scheduler. Default is 10.
        gamma (float): Multiplicative factor of learning rate decay. Default is 0.1.

    Attributes:
        id2label (dict): A dictionary mapping class IDs to class labels.
        num_classes (int): The number of classes.
        label2id (dict): A dictionary mapping class labels to class IDs.
        ignore_index (int): The index to ignore during training.
        model (SegformerForSemanticSegmentation): The Segformer model for semantic segmentation.
    """

    def __init__(self, id2label, metrics_interval=100, lr=2e-5, eps=1e-8, step_size=10, gamma=0.1):
        super(SegformerFinetuner, self).__init__()
        self.save_hyperparameters()
        self.id2label = id2label
        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.ignore_index = 0

        # Model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=True,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        # Metrics (ignore_index removed)
        self.train_iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes, ignore_index=0)
        self.train_dice = Dice(
            average='micro', num_classes=self.num_classes, ignore_index=0)
        self.val_iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes, ignore_index=0)
        self.val_dice = Dice(
            average='micro', num_classes=self.num_classes, ignore_index=0)
        self.test_iou = JaccardIndex(
            task='multiclass', num_classes=self.num_classes, ignore_index=0)
        self.test_dice = Dice(
            average='micro', num_classes=self.num_classes, ignore_index=0)

    def forward_pass(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        loss, logits = outputs['loss'], outputs['logits']

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],  # Match ground truth size
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        return loss, predicted

    def compute_and_log_metrics(self, predicted, masks, phase):
        # Flatten predicted and masks for multiclass confusion matrix
        predicted = predicted.view(-1)
        masks = masks.view(-1)

        # Mask out the ignore class (255) from both predicted and masks
        mask_ignore = (masks == 255)  # Exclude ignore (255) class
        predicted = predicted[~mask_ignore]
        masks = masks[~mask_ignore]

        print(f"Predicted shape: {predicted.shape}, Mask shape: {masks.shape}")
        print(f"Unique values in predicted: {torch.unique(predicted)}")
        print(f"Unique values in masks: {torch.unique(masks)}")

        # Compute IoU and Dice metrics
        if phase == "train":
            iou = self.train_iou(predicted, masks)
            dice = self.train_dice(predicted, masks)
        elif phase == "val":
            iou = self.val_iou(predicted, masks)
            dice = self.val_dice(predicted, masks)
        else:  # phase == "test"
            iou = self.test_iou(predicted, masks)
            dice = self.test_dice(predicted, masks)

        self.log(f"{phase}_iou", iou, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_dice", dice, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)
        self.compute_and_log_metrics(predicted, masks, phase="train")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)
        self.compute_and_log_metrics(predicted, masks, phase="val")
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        loss, predicted = self.forward_pass(images, masks)
        self.compute_and_log_metrics(predicted, masks, phase="test")
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr, eps=self.hparams.eps
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    def save_pretrained_model(self, path):
        self.model.save_pretrained(path)
