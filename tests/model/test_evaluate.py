import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from utils import Config, lc_id2label
from model.evaluate import evaluate


@pytest.fixture
def mock_config(tmp_path):
    """
    Mock configuration object.
    """
    config_dict = {
        "project": {
            "results_dir": str(tmp_path / "results"),
            "logs_dir": str(tmp_path / "logs"),
        }
    }
    results_dir = Path(config_dict["project"]["results_dir"])
    logs_dir = Path(config_dict["project"]["logs_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return Config.from_dict(config_dict)


@patch("utils.find_checkpoint")
@patch("data.loader.Loader")
@patch("model.lightning_model.SegformerFinetuner.load_from_checkpoint")
@patch("pytorch_lightning.Trainer")
@patch("utils.plot_confusion_matrix")
def test_evaluate_script(
    mock_plot_cm,
    mock_trainer,
    mock_load_model,
    mock_loader,
    mock_find_checkpoint,
    mock_config,
):
    # Mock dependencies
    mock_find_checkpoint.return_value = "/mock/path/to/checkpoint.ckpt"
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_test_results = {"ground_truths": [0, 1, 1], "predictions": [0, 1, 0]}
    mock_model.get_test_results.return_value = mock_test_results
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance

    # Run the script
    evaluate(config=mock_config, version="0")

    # Assertions
    mock_find_checkpoint.assert_called_once_with(mock_config, "0")
    mock_loader.assert_called_once_with(mock_config)
    mock_trainer.assert_called_once()
    mock_trainer_instance.test.assert_called_once_with(
        mock_model, mock_loader().get_dataloader("test"))
    mock_model.get_test_results.assert_called_once()

    # Correct label list from `lc_id2label`
    expected_labels = list(lc_id2label.values())
    save_path = Path(mock_config.project.results_dir) / \
        "version_0_confusion_matrix.png"

    # Fix the assertion
    mock_plot_cm.assert_called_once_with(
        [0, 1, 1], [0, 1, 0], expected_labels, save_path=save_path)
