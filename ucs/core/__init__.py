# __init__.py

from .errors import (
    BaseError,
    CheckpointDirectoryError,
    CheckpointNotFoundError,
    ConfigId2LabelError,
    LossIgnoreIndexError,
    LossWeightsSizeError,
    LossWeightsTypeError,
    MultipleCheckpointsError,
    NormalizeError,
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
