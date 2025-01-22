import pytest
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader
from data.loader import Loader


@pytest.fixture
def mock_config():
    config = Mock()
    config.dataset_path = "test_dataset"
    config.training.batch_size = 4
    config.training.num_workers = 2
    config.training.model_name = "b0"
    return config


@pytest.fixture
def mock_dataset():
    return {
        "train": [Mock() for _ in range(10)],
        "val": [Mock() for _ in range(5)],
        "test": [Mock() for _ in range(5)]
    }


@patch('data.loader.load_dataset')
def test_loader_initialization(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset

    # Adjusted initialization
    loader = Loader(
        dataset_path=mock_config.dataset_path,
        batch_size=mock_config.training.batch_size,
        num_workers=mock_config.training.num_workers,
        model_name=mock_config.training.model_name
    )

    assert loader.dataset == mock_dataset
    assert loader.batch_size == mock_config.training.batch_size
    assert loader.num_workers == mock_config.training.num_workers


@patch('data.loader.load_dataset')
def test_get_dataloader_train(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset

    # Adjusted initialization
    loader = Loader(
        dataset_path=mock_config.dataset_path,
        batch_size=mock_config.training.batch_size,
        num_workers=mock_config.training.num_workers,
        model_name=mock_config.training.model_name
    )

    dataloader = loader.get_dataloader("train")
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == mock_config.training.batch_size


@patch('data.loader.load_dataset')
def test_get_dataloader_val(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset

    # Adjusted initialization
    loader = Loader(
        dataset_path=mock_config.dataset_path,
        batch_size=mock_config.training.batch_size,
        num_workers=mock_config.training.num_workers,
        model_name=mock_config.training.model_name
    )

    dataloader = loader.get_dataloader("val")
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == mock_config.training.batch_size


@patch('data.loader.load_dataset')
def test_persistent_workers_setting(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset

    # Adjusted initialization
    loader = Loader(
        dataset_path=mock_config.dataset_path,
        batch_size=mock_config.training.batch_size,
        num_workers=mock_config.training.num_workers,
        model_name=mock_config.training.model_name
    )

    dataloader = loader.get_dataloader("train")
    assert dataloader.num_workers == mock_config.training.num_workers
