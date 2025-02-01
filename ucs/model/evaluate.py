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


def save_confusion_matrix(conf_matrix, metrics, labels, save_path, version):
    """
    Plot and save the confusion matrix with metrics.

    Args:
        conf_matrix: Confusion matrix as a NumPy array.
        metrics: Dictionary of evaluation metrics.
        labels: List of class labels for the confusion matrix.
        save_path: Path to save the confusion matrix plot.
    """
    from ucs.utils import save_confusion_matrix_plot
    # Save confusion matrix plot
    cm_save_path = Path(save_path) / f"version_{version}_confusion_matrix.png"

    print(f"Saving confusion matrix to: {cm_save_path}")
    save_confusion_matrix_plot(
        conf_matrix=conf_matrix,
        labels=labels,
        save_path=cm_save_path,
        metrics=metrics
    )
    print("Confusion matrix saved successfully.")


def evaluate(config, version=None):
    from pytorch_lightning import Trainer

    if not version:
        version = get_last_version(logs_dir=Path(config.directories.logs))

    # Initialize the data module
    datamodule = SegmentationDataModule(config.dataset)
    # Load the model from the checkpoint
    model = load_model_from_checkpoint(config, version)

    # Evaluate the model
    trainer = Trainer()
    tests = trainer.test(model, datamodule=datamodule)
    metrics = tests[0]
    conf_matrix = model.calculate_confusion_matrix()

    save_confusion_matrix(conf_matrix, metrics, list(
        config.training.id2label.values()), config.directories.results, version)
