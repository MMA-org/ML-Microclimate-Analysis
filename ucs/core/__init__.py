# __init__.py

from .errors import (
    BaseError,
    CheckpointNotFoundError,
    CheckpointDirectoryError,
    MultipleCheckpointsError,
    ConfigId2LabelError,
    NormalizeError,
    LossWeightsTypeError,
    LossWeightsSizeError,
    LossIgnoreIndexError,
)

# Define the public API
__all__ = [
    "BaseError",
    "CheckpointNotFoundError",
    "CheckpointDirectoryError",
    "MultipleCheckpointsError",
    "ConfigId2LabelError",
    "NormalizeError",
    "LossWeightsTypeError",
    "LossWeightsSizeError",
    "LossIgnoreIndexError",
]
