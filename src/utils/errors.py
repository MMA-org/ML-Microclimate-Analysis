import sys


class BaseError(Exception):
    """Base class for all project-specific errors."""
    exit_code = 1
    message = "An error occurred."

    def __init__(self, *args):
        self.args = args
        if args:
            self.message = self.message.format(*args)
        self.handle()

    def handle(self):
        """Prints the error message and exits with the specified code."""
        sys.stderr.write(f"Error: {self.message}\n")
        sys.exit(self.exit_code)


class CheckpointNotFoundError(BaseError):
    """Raised when no checkpoint files are found."""
    exit_code = 2
    message = "No checkpoint files found in directory: {}."


class CheckpointDirectoryError(BaseError):
    """Raised when the checkpoint directory is invalid or missing."""
    exit_code = 3
    message = "Checkpoint directory not found or invalid: {}."


class MultipleCheckpointsError(BaseError):
    """Raised when multiple checkpoint files are found."""
    exit_code = 4
    message = "Multiple checkpoint files found in directory: {}. Expected exactly one."


class ConfigId2LabelError(BaseError):
    """Raised when the 'id2label' mapping is missing from the configuration or dataset."""
    exit_code = 5
    message = "'id2label' mapping is missing in config. Please provide it to ensure correct label mapping."
