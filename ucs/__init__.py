from warnings import filterwarnings
from os import environ
# Disable Albumentations update check
environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

filterwarnings(
    "ignore",
    module="transformers.utils.deprecation",
    category=UserWarning
)
