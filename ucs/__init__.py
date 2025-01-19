import warnings
import os
# Disable Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

warnings.filterwarnings(
    "ignore",
    module="transformers.utils.deprecation",
    category=UserWarning
)
