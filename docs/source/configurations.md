# Configurations

This document provides an overview of the `config.yaml` file used for configuring the project and explains the available command-line options for overriding configurations.

## Overview of `config.yaml`

The `config.yaml` file organizes the project configuration into three main sections: `DATA`, `directories`, and `training`.

```{note}
The YAML file is **not required** for running the project with default settings. The project is designed to run seamlessly without a `config.yaml` file by using pre-defined default configuration values.
```

---

## Default configurations

### Data Configuration

The `DATA` section specifies the dataset and its properties:

```yaml
dataset:
  dataset_path: "erikpinhasov/landcover_dataset"
  batch_size: 16
  num_workers: 8
  do_reduce_labels: False
  pin_memory: True
```

**Details:**

- `dataset_path`: Path to the dataset. In this case, it references a Hugging Face dataset.
- `batch_size`: Number of samples processed per training batch.
- `num_workers`: Number of worker threads for data loading.
- `do_reduce_labels`: Whether to apply label reduction.
- `pin_memory`: If `True`, enables faster GPU transfer by pinning memory.
- `id2label`: Maps class indices to their corresponding labels.

---

### Project Configuration

The `directories` section defines directory paths for saving models, logs, and results:

```yaml
directories:
  models: models
  pretrained: models/pretrained_models
  logs: models/logs
  checkpoints: models/logs/checkpoints
  results: results
```

**Details:**

- `models`: Directory to save trained models.
- `pretrained`: Directory for storing pre-trained models.
- `logs`: Directory for logs.
- `checkpoints`: Directory for storing training checkpoints.
- `results`: Directory for evaluation results.

---

### Training Configuration

The `training` section provides options for training parameters:

```yaml
training:
  model_name: b0
  max_epochs: 50
  learning_rate: 2e-5
  weight_decay: 1e-3
  ignore_index: 0
  weighting_strategy: "raw"
  alpha: 0.7
  id2label:
    0: "background"
    1: "bareland"
    2: "rangeland"
    3: "developed space"
    4: "road"
    5: "tree"
    6: "water"
    7: "agriculture land"
    8: "buildings"
```

**Details:**

- `model_name`: Name of the SegFormer variant (e.g., `b0`).
- `max_epochs`: Maximum number of epochs for training.
- `learning_rate`: Learning rate for the optimizer.
- `weight_decay`: Strength of L2 regularization applied to the optimizer.
- `ignore_index`: Label index to ignore during training (`int` or `None`).
- `weighting_strategy`: Method for normalizing class weights (`"none"`, `"balanced"`, `"max"`, `"sum"`, `"raw"`).
- `alpha`: Cross-Entroy loss alpha weight parameter, beta = 1 - alpha.
- `id2label`: Mapping from class indices to class labels.

---

### Callbacks Configuration

The `callbacks` section defines parameters for managing callbacks during training:

```yaml
callbacks:
  early_stop_patience: 5
  early_stop_monitor: "val_loss"
  early_stop_mode: "min"
  save_model_monitor: "val_mean_iou"
  save_model_mode: "max"
```

**Details:**

- `early_stop_patience`: Number of epochs to wait for improvement before stopping training.
- `early_stop_monitor`: Metric to monitor for early stopping (e.g., `"val_loss"`).
- `early_stop_mode`: Direction of improvement (`"min"` for decreasing metrics, `"max"` for increasing metrics).
- `save_model_monitor`: Metric to track for saving the best model (e.g., `"val_mean_iou"`).
- `save_model_mode`: Direction of improvement (`"min"` for decreasing metrics, `"max"` for increasing metrics).

---

## Command-Line Options

Command-line options allow overriding `config.yaml` settings or default values. For example:

- Specify a custom configuration file:

  ```bash
  ucs --config path/to/config.yaml train
  ```

- Override specific parameters without a configuration file:
  ```bash
  ucs train --batch_size 32 --learning_rate 0.001
  ```

---

For more details, refer to the [Usage Guide](./usage.md).
