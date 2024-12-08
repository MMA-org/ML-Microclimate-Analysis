import argparse
from model.lightning_model import SegformerFinetuner
from data.loader import Loader
from utils import Config, lc_id2label, find_checkpoint, plot_confusion_matrix
from pytorch_lightning import Trainer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer")
    parser.add_argument(
        "--config", type=str, default="default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--version", type=str, default="version_0", help="Version of model to evaluate"
    )
    args = parser.parse_args()

    print(f"Evaluating model version: {args.version}")

    # Load the configuration (directories are automatically created)
    config = Config(args.config)

    # Locate the checkpoint
    checkpoint = find_checkpoint(config, args.version)
    print(f"Using checkpoint: {checkpoint}")

    # Prepare the test dataloader
    loader = Loader(config)
    test_loader = loader.get_dataloader("test")

    # Load the model from the checkpoint
    model = SegformerFinetuner.load_from_checkpoint(
        checkpoint_path=checkpoint,
        id2label=lc_id2label,
    )

    # Evaluate the model
    trainer = Trainer()
    trainer.test(model, test_loader)

    # Collect predictions and ground truths
    y_true = model.test_ground_truths
    y_pred = model.test_predictions

    # Plot and save the confusion matrix
    results_dir = Path(config.project.results_dir)
    cm_save_path = results_dir / f"version_{args.version}_confusion_matrix.png"
    labels = list(lc_id2label.values())
    plot_confusion_matrix(y_true, y_pred, labels, save_path=cm_save_path)

    print(f"Confusion matrix saved to {cm_save_path}")


if __name__ == "__main__":
    main()
