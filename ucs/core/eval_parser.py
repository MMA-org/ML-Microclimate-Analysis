from core.parser import ArgParser


class EvalParser:
    def __init__(self):
        self.parser = ArgParser(
            usage="ucs evaliate [--help|-h] [options]"
        )
        self.add_arguments()

    def add_arguments(self):
        # Add evaluation-specific arguments
        self.parser.add_argument(
            "-v", "--version", type=int,
            help="Model version from checkpoints to evaluate."
        )

    def get_parser(self):
        return self.parser
