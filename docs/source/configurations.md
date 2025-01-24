# Configurations

This document provides an overview of the `config.yaml` file used for configuring the project and explains the available command-line options for overriding configurations.

## Overview of `config.yaml`

The `config.yaml` file organizes the project configuration into three main sections: `DATA`, `project`, and `training`.

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

- `dataset_path`: Path to the dataset. In this case, it references a Hugging Face dataset.
- `id2label`: Maps class indices to their corresponding labels.

---

### Project Configuration

The `project` section defines directory paths for saving models, logs, and results:

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
  batch_size: 16
  max_epochs: 50
  num_workers: 8
  learning_rate: 2e-5
  weight_decay: 1e-4
```

**Details:**

- `batch_size`: Number of samples processed per training batch.
- `max_epochs`: Maximum number of epochs for training.
- `num_workers`: Number of worker threads for data loading.
- `learning_rate`: Learning rate for the optimizer.
- `weight_decay`: Strength of L2 regularization applied to the optimizer.
- `model_name`: Name of the model architecture (e.g., `b4`).

---

### Callbacks Configuration

The `callbacks` section in the configuration defines parameters for managing callbacks during training:

```yaml
callbacks:
  early_stop:
    patience: 5
    monitor: "val_loss"
    mode: "min"
  save_model:
    monitor: "val_mean_iou"
    mode: "max"
```

**Details**

- `early_stop.patience`: Number of epochs to wait for improvement before stopping training.
- `early_stop.monitor`: Metric to monitor for early stopping (e.g., `"val_loss"`).
- `early_stop.mode`: Direction of improvement (`"min"` for decreasing metrics, `"max"` for increasing metrics).
- `save_model.monitor`: Metric to track for saving the best model (e.g., `"val_mean_iou"`).
- `save_model.mode`: Direction of improvement (`"min"` for decreasing metrics, `"max"` for increasing metrics).

---

### Loss Configuration

The loss function combines `Cross-Entropy` Loss and `Dice` Loss. Below are the configurable parameters:

```yaml
loss:
  ignore_index: 0
  weighting_strategy: "raw"
  alpha: 0.5
  beta: 0.5
```

**Details:**

- `ignore_index`: Index for the class to ignore during loss calculation. Use `None` if no class should be ignored.
- `weighting_strategy`: Method for normalizing class weights.
  - `max`: Scales weights relative to the maximum weight.
  - `sum`: Scales weights so their sum equals 1.
  - `raw`: Uses raw, unnormalized weights
  - `balanced`: Adjusts weights to ensure equal contribution from all classes.
  - `none`: Do not normalize weights
- `alpha`: Weight for the Cross-Entropy Loss component in the combined loss.
- `beta`: Weight for the Dice Loss component in the combined loss.

---

## Command-Line Options

Command-line options allow overriding `config.yaml` settings or default values. For example:

- Specify a custom configuration file:

  ```bash
  ucs train --config path/to/config.yaml
  ```

- Override specific parameters without a configuration file:
  ```bash
  ucs train --batch_size 32 --lr 0.001
  ```

---

For more details, refer to the [Usage Guide](./usage.md).
