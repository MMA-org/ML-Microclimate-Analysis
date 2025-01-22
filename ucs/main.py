
from core.parser import ArgParser
from argparse import RawTextHelpFormatter
from utils.config import Config


def create_common_parser():
    common_parser = ArgParser(add_help=False)
    common_parser.add_argument("--dataset_path", type=str, metavar="[path]",
                               help="Path to huggingface or local dataset.")
    common_parser.add_argument("--id2label", type=str, metavar="[json]",
                               help="ID to label mapping as a JSON string.")

    # Project arguments
    common_parser.add_argument("--models_dir", type=str, metavar="[path]",
                               help="Directory to save models.")
    common_parser.add_argument("--pretrained_dir", type=str, metavar="[path]",
                               help="Directory for pretrained models.")
    common_parser.add_argument("--logs_dir", type=str, metavar="[path]",
                               help="Directory for logs.")
    common_parser.add_argument("--results_dir", type=str, metavar="[path]",
                               help="Directory for results.")
    return common_parser


def create_train_parser(subparsers, common_parser):
    """
    Create the train subparser.
    """
    train_parser = subparsers.add_parser(
        "train",
        help="Train the model.",
        parents=[common_parser],
        formatter_class=RawTextHelpFormatter
    )
    # Add training-specific arguments
    train_parser.add_argument(
        "--resume", type=int, help="Specify the model version number to resume training from.")
    train_parser.add_argument(
        "--batch_size", type=int,  help="Batch size for training.")
    train_parser.add_argument(
        "--max_epochs", type=int,  help="Maximum number of epochs.")
    train_parser.add_argument(
        "--lr", type=float,  help="Learning rate.")
    train_parser.add_argument(
        "--dropout", type=float,  help="Probability for dropout.")
    train_parser.add_argument(
        "--weight_decay", type=float,  help="Weight decay.")
    train_parser.add_argument(
        "--model_name", type=str, choices=["b0", "b1", "b2", "b3", "b4"], metavar="b1-b4",  help="Model name.")
    train_parser.add_argument(
        "--stop_patience", type=int,  help="Early stopping patience.")
    train_parser.add_argument("--checkpoints_dir", type=str, metavar="[path]",
                              help="Path to checkpoints directiory.")
    # Add focal loss arguments
    train_parser.add_argument(
        "--alpha", type=float,  help="loss alpha.")
    train_parser.add_argument(
        "--beta", type=float,  help="loss beta.")
    train_parser.add_argument(
        "--ignore_index", type=int,  help="loss function ignore index.")
    train_parser.add_argument(
        "--class_weights",
        type=ArgParser.str_to_bool,  # Use the custom boolean parser
        help="Use class weights for loss function (true/false)."
    )

    train_parser.add_argument("--normalize", type=str, choices=["max", "sum", "none", "balanced"],
                              metavar="[max | sum | none | balanced]", help="Normalization method for class weights.")
    return train_parser


def create_evaluate_parser(subparsers, common_parser):
    """
    Create the evaluate subparser.
    """
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the model.",
        parents=[common_parser],
        formatter_class=RawTextHelpFormatter
    )
    evaluate_parser.add_argument(
        "-v", "--version", type=int,
        help="Model version from checkpoints to evaluate."
    )
    return evaluate_parser


def parse_args():
    """
    Parse top-level command-line arguments and subcommands (train/evaluate).
    """
    # Main parser setup
    # Main parser
    parser = ArgParser(
        formatter_class=RawTextHelpFormatter,
        description=(
            "LandCover Model:\n"
            "This script allows you to train or evaluate a SegFormer model for land-cover semantic segmentation tasks.\n"
        ),
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="Path to config yaml file. (default: config.yaml)", metavar="[path]")

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command", required=False, title="Commands")

    # Common parser for shared arguments
    common_parser = create_common_parser()

    # Add train and evaluate subparsers
    create_train_parser(subparsers, common_parser)
    create_evaluate_parser(subparsers, common_parser)

    return parser.parse_args()


def handle_arparse(args):
    """
    Handle and parse arguments, extracting command details and configuration path.
    """

    args_dict = vars(args)
    config_path = args.config
    command = args.command
    args_dict.pop("config", None)
    args_dict.pop("command", None)
    return args_dict, config_path, command


def main():
    """
    Entry point for the LandCover Model script. Parses arguments, loads configuration,
    and dispatches train or evaluate based on the selected command.
    """
    args = parse_args()
    args_dict, config_path, command = handle_arparse(args)

    # Load configuration
    config = Config(config_path=config_path)
    config.load_from_args(args_dict)
    config.__create_directories__()

    # Dispatch the appropriate subcommand
    if command == "train":
        from model.train import train
        train(config, resume_version=args.resume)
    elif command == "evaluate":
        from model.evaluate import evaluate
        evaluate(config, version=args.version)


if __name__ == "__main__":
    main()
