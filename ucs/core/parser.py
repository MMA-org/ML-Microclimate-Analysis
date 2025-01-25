import argparse


class ArgParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser to automatically set metavar based on the argument type.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('formatter_class', argparse.RawTextHelpFormatter)
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        """
        Override the add_argument method to set metavar based on the argument type.
        """
        if 'metavar' not in kwargs and 'type' in kwargs:
            arg_type = kwargs['type']
            if arg_type is int:
                kwargs['metavar'] = '[int]'
            elif arg_type is float:
                kwargs['metavar'] = '[float]'
            elif arg_type is str:
                kwargs['metavar'] = '[path]' if 'path' in args[0] or 'dir' in args[0] else 'str'
            elif arg_type == bool:
                kwargs['type'] = self.str_to_bool
                kwargs['metavar'] = '[true | false]'

        # Call the original add_argument method
        super().add_argument(*args, **kwargs)

    @staticmethod
    def str_to_bool(value):
        """
        Convert a string value to a boolean. Supports values like 'true', 'false', '1', '0', 'yes', 'no'.
        """
        if value is None:
            return None
        elif value.lower() in ("no", "false", "f", "n", "0"):
            return False
        elif value.lower() in ("yes", "true", "t", "y", "1"):
            return True
        else:
            raise argparse.ArgumentTypeError(
                "Boolean value expected (e.g., True|False).")

    def parse_args(self, args=None, namespace=None):
        """
        Override the parse_args method to handle missing subcommands.
        """
        if args is None:
            args = super().parse_args(namespace=namespace)

        # If no subcommand is provided, print help and exit
        if not hasattr(args, "command") or args.command is None:
            self.print_help()
            self.exit(1)

        return args
