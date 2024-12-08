

def evaluate(config, version="0"):
    from model.lightning_model import SegformerFinetuner
    from data.loader import Loader
    from utils import lc_id2label, find_checkpoint, plot_confusion_matrix
    from pytorch_lightning import Trainer
    from pathlib import Path
    print(f"Evaluating model version: version_{version}")

    # Locate the checkpoint
    checkpoint = find_checkpoint(config, version)
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
    cm_save_path = results_dir / f"version_{version}_confusion_matrix.png"
    labels = list(lc_id2label.values())
    plot_confusion_matrix(y_true, y_pred, labels, save_path=cm_save_path)

    print(f"Confusion matrix saved to {cm_save_path}")
