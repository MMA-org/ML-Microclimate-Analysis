# Configurations

This document provides an overview of the `config.yaml` file used for configuring the project and explains the available command-line options for overriding configurations.

## Overview of `config.yaml`

The `config.yaml` file organizes the project configuration into three main sections: `DATA`, `project`, and `training`.

---

> **_NOTE_**  
> The YAML file is **not required** for running the project with default settings. The project is designed to run seamlessly without a `config.yaml` file by using pre-defined default configuration values.

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
project:
  models_dir: models
  pretrained_dir: models/pretrained_models
  logs_dir: models/logs
  results_dir: results
```

**Details:**

- `models_dir`: Directory to save trained models.
- `pretrained_dir`: Directory for storing pre-trained models.
- `logs_dir`: Directory for logs.
- `results_dir`: Directory for evaluation results.

---

### Training Configuration

The `training` section provides options for training parameters:

```yaml
training:
  batch_size: 16
  max_epochs: 50
  num_workers: 8
  log_every_n_steps: 10
  learning_rate: 1e-3
  model_name: b4
  early_stop:
    patience: 10

  focal_loss:
    weights:
      class_weights: True # bool (default True)
      normalize: "balanced" # max | sum | balanced (default 'balanced')
    gamma: 2.0 # float
    alpha: None # None | float
    ignore_index: 0 # None | int
```

**Details:**

- `batch_size`: Number of samples processed per training batch.
- `max_epochs`: Maximum number of epochs for training.
- `num_workers`: Number of worker threads for data loading.
- `log_every_n_steps`: Logging frequency during training.
- `learning_rate`: Learning rate for the optimizer.
- `model_name`: Name of the model architecture (e.g., `b4`).
- `early_stop.patience`: Number of epochs to wait for improvement before stopping training.
- `focal_loss`: Configuration for the focal loss function, with options for class weights, gamma, and alpha.

---

## Command-Line Options

Command-line options allow overriding `config.yaml` settings or default values. For example:

- Specify a custom configuration file:

  ```bash
  lcm train --config path/to/config.yaml
  ```

- Override specific parameters without a configuration file:
  ```bash
  lcm train --batch_size 32 --lr 0.001
  ```

Refer to the [Usage Guide](./usage.md) for additional examples.

---

For more details, refer to the [Usage Guide](./usage.md).
