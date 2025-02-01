import logging
import warnings
import os

# Disable Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning,
                        module="transformers.utils.deprecation")

# Reduce logging verbosity from transformers to avoid unnecessary messages
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
