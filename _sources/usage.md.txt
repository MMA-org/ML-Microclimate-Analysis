# Usage

This document provides guidance on how to use the **Urban Climate Segmentation Model** repository for training, evaluating, and managing outputs efficiently.

---

## Default Configurations

The project is designed to run seamlessly even if `config.yaml` does not exist. When no configuration file is provided, it automatically uses default settings. If a `config.yaml` file exists, it will be loaded automatically. Refer to the [Configurations Guide](./configurations.md) for details.

---

## Training the Model

Use the `ucs train` command to start training. Examples:

1. **Default Training**:

   ```bash
   ucs train
   ```

2. **With Custom Configurations**:

   ```bash
   ucs train --config path/to/config.yaml
   ```

3. **Command-Line Overrides**:

   ```bash
   ucs train --batch_size 32 --lr 0.001
   ```

### Training Outputs

During training, the following outputs are generated:

- **Checkpoints**:
  Saved in `models/logs/checkpoints/`. This includes the best model checkpoints based on validation performance.
- **Pretrained Models**:
  The pretrained BERT-based transformer model from Hugging Face is saved in `models/pretrained_models/`.
- **Lightning Logs**:
  Training logs for TensorBoard are stored in `models/logs/lightning_logs/`. These logs include metrics and other details for visualization.

To view all available training options, run:

```bash
ucs train -h
```

---

## Evaluating the Model

Evaluate a specific version of the model and save results, including a confusion matrix, to the `results/` directory.

### Running Evaluation

1. **Default Evaluation**:

   ```bash
   ucs evaluate
   ```

2. **Evaluate Specific Version**:

   ```bash
   ucs evaluate -v <version>
   ```

   Example:

   ```bash
   ucs evaluate -v 5
   ```

### Evaluation Outputs

- **Confusion Matrix**:
  Saved in the `results/` directory as:
  ```bash
  results/version_<version>_confusion_matrix.png
  ```

---

## Folder Structure and Outputs

During training and evaluation, outputs are organized as follows:

- **`models/`**: Root folder for all model-related outputs.
  - **`models/logs/`**:
    - **`models/logs/checkpoints/`**: Contains the best model checkpoints.
    - **`models/logs/lightning_logs/`**: Training logs for TensorBoard.
  - **`models/pretrained_models/`**:
    - Contains pretrained models used for initializing training.
- **`results/`**:
  - Contains evaluation results, including confusion matrices saved as:
    - `results/version_<version>_confusion_matrix.png`

---

For more details, refer to the [Configurations Guide](./configurations.md).
