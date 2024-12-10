import pytest
from unittest.mock import MagicMock, patch
from utils import Config, lc_id2label
from model.train import train


@pytest.fixture
def mock_config(tmp_path):
    """
    Fixture for a mock configuration object.
    """
    config_dict = {
        "project": {
            "logs_dir": str(tmp_path / "logs"),
            "pretrained_dir": str(tmp_path / "pretrained_models"),
        },
        "training": {
            "do_class_weight": True,
            "model_name": "b1",
            "patience": 5,
            "max_epochs": 10,
            "log_every_n_steps": 10,
        },
    }
    return Config.from_dict(config_dict)


# Update to match the actual import path in train.py
@patch("data.loader.Loader")
# Match the actual import path
@patch("model.lightning_model.SegformerFinetuner")
@patch("pytorch_lightning.loggers.TensorBoardLogger")  # Match the import path
@patch("pytorch_lightning.callbacks.ModelCheckpoint")  # Match the import path
@patch("pytorch_lightning.callbacks.EarlyStopping")  # Match the import path
# Match the import path
@patch("utils.save_pretrained_callback.SavePretrainedCallback")
@patch("pytorch_lightning.Trainer")  # Match the import path
@patch("utils.metrics.compute_class_weights")  # Match the import path
def test_train(
    mock_compute_class_weights,
    mock_trainer,
    mock_save_pretrained_callback,
    mock_early_stopping,
    mock_checkpoint,
    mock_logger,
    mock_finetuner,
    mock_loader,
    mock_config,
):
    # Mock dependencies
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.get_dataloader.side_effect = [
        mock_train_loader, mock_val_loader
    ]

    mock_model = MagicMock()
    mock_finetuner.return_value = mock_model

    mock_class_weights = MagicMock()
    mock_compute_class_weights.return_value = mock_class_weights

    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance

    # Call the train function
    train(mock_config)

    # Assertions
    mock_loader.assert_called_once_with(mock_config)
    mock_loader_instance.get_dataloader.assert_any_call("train", shuffle=True)
    mock_loader_instance.get_dataloader.assert_any_call("validation")

    # Update expected arguments to match the actual function call
    mock_compute_class_weights.assert_called_once_with(
        mock_train_loader, lc_id2label
    )

    mock_finetuner.assert_called_once_with(
        id2label=lc_id2label,
        model_name="b1",
        class_weight=mock_class_weights,
    )

    mock_logger.assert_called_once_with(
        save_dir=mock_config.project.logs_dir, default_hp_metric=False
    )

    mock_checkpoint.assert_called_once()
    mock_early_stopping.assert_called_once()
    mock_save_pretrained_callback.assert_called_once()

    mock_trainer.assert_called_once()
    mock_trainer_instance.fit.assert_called_once_with(
        mock_model, mock_train_loader, mock_val_loader, ckpt_path=None
    )
