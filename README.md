# Urban Climate Segmentation Model

Welcome to the **Urban Climate Segmentation Model** repository. This project is dedicated to advancing semantic segmentation tasks with a specific focus on urban climate analysis. Our model has been fine-tuned using state-of-the-art datasets to explore land cover and building detection in urban environments.

---

## Repository Structure and Navigation

This repository is organized into several key sections for ease of use:

### Documentation

- [Installation Guide](./docs/source/installation.md): Instructions to set up and install the project on your local machine.
- [Configuration Guide](./docs/source/configurations.md): Details on project configuration, including examples and use cases.
- [Dataset Documentation](./docs/source/Dataset.md): Comprehensive information on the datasets used in the project.
- [Usage Guide](./docs/source/usage.md): Step-by-step instructions for training, evaluation, and model usage.
- [Project Documentation](https://mma-org.github.io/ML-Microclimate-Analysis/index.html): Centralized project documentation hosted online.

### Source Code

- **`src/`**: Core implementation of the project, including:
  - **`core/`**: Core functionality and model pipelines.
  - **`data/`**: Data loaders, preprocessing pipelines, and dataset utilities.
  - **`model/`**: Model architectures and training scripts.
  - **`utils/`**: Utility scripts for data preprocessing, model evaluation, and more.

### Tests

- **`tests/`**: Unit and integration tests to ensure the reliability of the project.

### Notebooks

- **`notebooks/`**: Jupyter notebooks for exploratory data analysis, visualizations, and model experimentation.

---

## Datasets

This repository leverages high-quality, publicly available datasets for training and evaluation. For detailed information on these datasets, including their structure and applications, please refer to the [Datasets Documentation](./docs/source/Dataset.md).

### Key Dataset

- **[LandCover Urban/Rural Climate Dataset](https://huggingface.co/datasets/erikpinhasov/landcover_dataset)**:
  - A comprehensive dataset for land cover classification in urban and rural regions, featuring annotations for various land types such as buildings, roads, and water bodies.

---

## Usage

This repository provides a seamless interface for training and evaluating segmentation models. Follow these steps to get started:

### Training

To train the model with default configurations, run:

```bash
lcm train
```

For custom configurations, provide a `config.yaml` file or override specific parameters via the command line:

```bash
lcm train --config path/to/config.yaml
lcm train --batch_size 32 --lr 0.001
```

### Evaluation

Evaluate a trained model and save the confusion matrix in the `results/` directory:

```bash
lcm evaluate -v <version>
```

For available options, run:

```bash
lcm train -h
lcm evaluate -h
```

For more details, refer to the [Usage Guide](./docs/source/usage.md).

---

### About the Project

The Urban Climate Segmentation Model is designed to support research and applications in urban planning, environmental monitoring, and climate impact studies. By providing robust tools for land cover segmentation, this project contributes to the growing body of knowledge addressing urban climate challenges.

For more information, visit the [project documentation](https://mma-org.github.io/ML-Microclimate-Analysis/index.html).
