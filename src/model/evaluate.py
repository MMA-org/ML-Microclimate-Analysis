import argparse
from model.lightning_model import SegformerFinetuner
from src.data.loader import Loader
from utils import Config, lc_id2label, find_checkpoint, plot_confusion_matrix
from pytorch_lightning import Trainer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer")
    parser.add_argument(
        "--config", type=str, default="default.yaml", help="Path to config file")
    parser.add_argument("--version", type=str, default="version_0",
                        help="Version of model to evaluate")

    args = parser.parse_args()
    print(f"Evaluate {args.version} of model.")
    # Load the configuration
    config = Config(args.config)

    # Access paths and settings from the config
    checkpoint = find_checkpoint(config, args.version)
    print(f"Using checkpoint: {checkpoint}")

    # Prepare test dataloader
    loader = Loader(config)
    test_loader = loader.get_dataloader("test")

    # Load model from checkpoint
    model = SegformerFinetuner.load_from_checkpoint(
        checkpoint_path=checkpoint,
        id2label=lc_id2label
    )

    # Evaluate
    trainer = Trainer()
    trainer.test(model, test_loader)

    # Collect predictions and ground truths
    y_true = model.test_ground_truths
    y_pred = model.test_predictions

    # Create result dir
    results_dir = Path(config.project.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    labels = list(lc_id2label.values())  # Use class labels
    cm_save_path = results_dir / f"{args.version}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, labels, save_path=cm_save_path)


if __name__ == "__main__":
    main()
