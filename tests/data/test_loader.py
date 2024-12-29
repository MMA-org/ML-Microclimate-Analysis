import pytest
from unittest.mock import Mock, patch
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
    mock_data = {
        "train": [Mock() for _ in range(10)],
        "val": [Mock() for _ in range(5)],
        "test": [Mock() for _ in range(5)]
    }
    return mock_data


@patch('data.loader.load_dataset')
def test_loader_initialization(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset
    loader = Loader(mock_config)
    assert loader.dataset == mock_dataset
    assert loader.batch_size == mock_config.training.batch_size
    assert loader.num_workers == mock_config.training.num_workers


@patch('data.loader.load_dataset')
def test_get_transforms(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset
    loader = Loader(mock_config)
    transforms = loader.get_transforms()
    assert isinstance(transforms, A.Compose)


@patch('data.loader.load_dataset')
def test_get_dataloader_train(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset
    loader = Loader(mock_config)
    dataloader = loader.get_dataloader('train', shuffle=True)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == mock_config.training.batch_size


@patch('data.loader.load_dataset')
def test_get_dataloader_val(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset
    loader = Loader(mock_config)
    dataloader = loader.get_dataloader('val', shuffle=False)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == mock_config.training.batch_size


@patch('data.loader.load_dataset')
def test_persistent_workers_setting(mock_load_dataset, mock_config, mock_dataset):
    mock_load_dataset.return_value = mock_dataset
    loader = Loader(mock_config)
    dataloader = loader.get_dataloader('train', shuffle=True)
    assert dataloader.num_workers == mock_config.training.num_workers
