from core.parser import ArgParser


class TrainParser:
    def __init__(self):
        self.parser = ArgParser(
            usage="mc train [--help|-h] [options]"
        )
        self.add_arguments()

    def add_arguments(self):

        # Add training-specific arguments
        self.parser.add_argument(
            "--resume", type=int, help="Specify the model version number to resume training from.")
        self.parser.add_argument(
            "--batch_size", type=int,  help="Batch size for training.")
        self.parser.add_argument(
            "--max_epochs", type=int,  help="Maximum number of epochs.")
        self.parser.add_argument(
            "--log_step", type=int,  help="Log every n steps.")
        self.parser.add_argument(
            "--lr", type=float,  help="Learning rate.")
        self.parser.add_argument(
            "--model_name", type=str, choices=["b0", "b1", "b2", "b3", "b4"], metavar="b1-b4",  help="Model name.")
        self.parser.add_argument(
            "--stop_patience", type=int,  help="Early stopping patience.")
        self.parser.add_argument("--checkpoints_dir", type=str, metavar="[path]",
                                 help="Path to checkpoints directiory.")
        # Add focal loss arguments
        self.parser.add_argument(
            "--alpha", type=float,  help="loss alpha.")
        self.parser.add_argument(
            "--beta", type=float,  help="loss beta.")
        self.parser.add_argument(
            "--ignore_index", type=int,  help="loss function ignore index.")
        self.parser.add_argument(
            "--class_weights", type=ArgParser.str_to_bool, help="Use class weights for loss function.")
        self.parser.add_argument("--normalize", type=str, choices=["max", "sum", "none", "balanced"],
                                 metavar="[max | sum | none | balanced]", help="Normalization method for class weights.")

    def get_parser(self):
        return self.parser
