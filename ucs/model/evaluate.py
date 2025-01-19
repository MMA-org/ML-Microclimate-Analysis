

def evaluate(config, version=None):
    from model.lightning_model import SegformerFinetuner

    from data.loader import Loader
    from utils import find_checkpoint, save_confusion_matrix_plot, get_last_version
    from pytorch_lightning import Trainer
    from pathlib import Path
    print(f"Evaluating model version: version_{version}")
    id2label = config.dataset.id2label
    # Locate the checkpoint
    if not version:
        version = get_last_version(logs_dir=Path(config.directories.logs))
    checkpoint = find_checkpoint(config, version)
    print(f"Using checkpoint: {checkpoint}")

    # Prepare the test dataloader
    loader = Loader(config)
    test_loader = loader.get_dataloader("test")

    # Load the model from the checkpoint
    model = SegformerFinetuner.load_from_checkpoint(
        checkpoint_path=checkpoint,
        id2label=id2label,
    )

    # Evaluate the model
    trainer = Trainer()
    tests = trainer.test(model, test_loader)
    print("Create Confusion matrix")
    tests_metrics_results = tests[0]

    test_results = model.get_test_results()
    y_true = test_results["ground_truths"]
    y_pred = test_results["predictions"]

    # Save confusion matrix
    results = Path(config.directories.results)
    cm_save_path = results / f"version_{version}_confusion_matrix.png"
    labels = list(id2label.keys())

    # Plot and save the confusion matrix
    save_confusion_matrix_plot(
        y_true, y_pred, labels, save_path=cm_save_path, metrics=tests_metrics_results)

    print(f"Confusion matrix saved to {cm_save_path}")
