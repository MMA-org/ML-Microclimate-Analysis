# Variables
PYTHON = python
SRC_DIR = src
SCRIPTS_DIR = scripts
DATA_DIR = ~/.cache/huggingface/datasets
CHECKPOINT_DIR = checkpoints
MODELS_DIR = models
RESULTS_DIR = results
export NO_ALBUMENTATIONS_UPDATE=1
# Default target
.DEFAULT_GOAL := help

# Rules
help:  ## Show help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean:  ## Clean up temporary files
	@echo "Cleaning up..."
	rm -rf $(MODELS_DIR) $(RESULTS_DIR) $(CHECKPOINT_DIR)

download_data:  ## Download or preprocess data
	@echo "Downloading or preprocessing data..."
	$(PYTHON) $(SRC_DIR)/data/download_data.py

train:  ## Train the model
	@echo "Training the model..."
	$(PYTHON) $(SRC_DIR)/model/train.py

evaluate:  ## Evaluate the model
	@echo "Evaluating the model..."
	$(PYTHON) $(SRC_DIR)/model/evaluate.py

visualize:  ## Visualize results
	@echo "Visualizing results..."
	$(PYTHON) $(SRC_DIR)/model/visualize.py

run_pipeline: setup download_data train evaluate visualize  ## Run the entire pipeline
	@echo "Running the full ML pipeline..."

.PHONY: help setup clean download_data train evaluate visualize run_pipeline
