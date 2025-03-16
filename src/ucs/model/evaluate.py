from pathlib import Path

from ucs.data.data_module import SegmentationDataModule
from ucs.utils import find_checkpoint, get_last_version


def load_model_from_checkpoint(config, version):
    """
    Load the model from a specific checkpoint.

    Args:
        config: Configuration object containing directories and model settings.
        version: Model version to locate the appropriate checkpoint.

    Returns:
        A SegformerFinetuner instance loaded from the checkpoint.
    """
    from ucs.model.lightning_model import SegformerFinetuner

    checkpoint = find_checkpoint(config.directories.checkpoints, version)

    # Load the model
    print(f"Loading model from checkpoint: {checkpoint}")
    return SegformerFinetuner.load_from_checkpoint(
        checkpoint_path=checkpoint,
    )


def evaluate(config, version=None):
    from pytorch_lightning import Trainer

    if not version:
        version = get_last_version(
            logs_dir=Path(config.directories.logs),
            project_name=config.loggers.project_name,
        )

    # Initialize the data module
    datamodule = SegmentationDataModule(config.dataset)
    # Load the model from the checkpoint
    model = load_model_from_checkpoint(config, version)

    # Evaluate the model
    trainer = Trainer()
    trainer.test(model, datamodule=datamodule)
    cm_save_path = (
        Path(config.directories.results) / f"version_{version}_confusion_matrix.png"
    )
    print(f"Saving confusion matrix to: {cm_save_path}")
    model.calculate_confusion_matrix(cm_save_path)
    print("Confusion matrix saved successfully.")
