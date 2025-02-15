import logging
import os
import warnings

from pytorch_lightning import seed_everything

# Disable Albumentations update check
seed_everything(42, workers=True, verbose=False)
os.environ["WANDB_SILENT"] = "true"  # Prevent W&B from printing to console
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Suppress specific warnings from transformers
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.utils.deprecation"
)

# Reduce logging verbosity from transformers to avoid unnecessary messages
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
