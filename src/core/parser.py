import argparse


class ArgParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser to automatically set metavar to the type of the argument.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('formatter_class', argparse.RawTextHelpFormatter)
        super().__init__(*args, **kwargs)
        self.add_common_arguments()

    def add_common_arguments(self):
        """
        Add common arguments shared by Train and Eval parsers.
        """
        # Dataset arguments
        self.add_argument("--dataset_path", type=str, metavar="[path]",
                          help="path to huggingface or local dataset.")
        self.add_argument("--id2label", type=str, metavar="[json]",
                          help="ID to label mapping as a JSON string.")

        # Project arguments
        self.add_argument("--models_dir", type=str, metavar="[path]",
                          help="Directory to save models.")
        self.add_argument("--pretrained_dir", type=str, metavar="[path]",
                          help="Directory for pretrained models.")
        self.add_argument("--logs_dir", type=str, metavar="[path]",
                          help="Directory for logs.")
        self.add_argument("--results_dir", type=str, metavar="[path]",
                          help="Directory for results.")

    def add_argument(self, *args, **kwargs):
        """
        Override the add_argument method to set metavar to the type of the argument.
        """

        if 'metavar' not in kwargs and 'type' in kwargs:
            arg_type = kwargs['type']
            if arg_type is int:
                kwargs['metavar'] = '[int]'
            elif arg_type is float:
                kwargs['metavar'] = '[float]'
            elif arg_type is str:
                kwargs['metavar'] = '[path]' if 'path' in args[0] or 'dir' in args[0] else 'str'
            elif arg_type == self.str_to_bool:
                kwargs['metavar'] = '[true | false]'

        # Call the original add_argument method
        super().add_argument(*args, **kwargs)

    @staticmethod
    def str_to_bool(value):
        """
        Convert a string value to a boolean.
        Supports values like 'true', 'false', '1', '0', 'yes', 'no'.
        """
        if isinstance(value, bool):
            return value
        if value.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif value.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError(
                "Boolean value expected (e.g., True|False).")
