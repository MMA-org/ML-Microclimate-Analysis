from argparse import RawTextHelpFormatter

from ucs.core.parser import ArgParser


def create_common_parser():
    common_parser = ArgParser(add_help=False)
    common_parser.add_argument(
        "--dataset_path",
        type=str,
        metavar="[path]",
        help="Path to huggingface or local dataset.",
    )
    common_parser.add_argument(
        "--id2label",
        type=str,
        metavar="[json]",
        help="ID to label mapping as a JSON string.",
    )

    # Project arguments
    common_parser.add_argument(
        "--models_dir", type=str, metavar="[path]", help="Directory to save models."
    )
    common_parser.add_argument(
        "--pretrained_dir",
        type=str,
        metavar="[path]",
        help="Directory for pretrained models.",
    )
    common_parser.add_argument(
        "--logs_dir", type=str, metavar="[path]", help="Directory for logs."
    )
    common_parser.add_argument(
        "--results_dir", type=str, metavar="[path]", help="Directory for results."
    )
    return common_parser


def create_train_parser(subparsers, common_parser):
    """
    Create the train subparser.
    """
    train_parser = subparsers.add_parser(
        "train",
        help="Train the model.",
        parents=[common_parser],
        formatter_class=RawTextHelpFormatter,
    )
    # Add training-specific arguments
    train_parser.add_argument(
        "--resume",
        type=int,
        help="Specify the model version number to resume training from.",
    )
    train_parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    train_parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs."
    )
    train_parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    train_parser.add_argument("--weight_decay", type=float, help="Weight decay.")
    train_parser.add_argument(
        "--model_name",
        type=str,
        choices=["b0", "b1", "b2", "b3", "b4"],
        metavar="b1-b4",
        help="Model name.",
    )
    train_parser.add_argument(
        "--early_stop_patience", type=int, help="Early stopping callback patience."
    )
    train_parser.add_argument(
        "--early_stop_monitor",
        type=str,
        help="Early stopping callback metrics to monitor.",
    )
    train_parser.add_argument(
        "--early_stop_mode",
        type=str,
        choices=["min", "max"],
        help="Early stopping callback metrics mode for metrics.",
    )
    train_parser.add_argument(
        "--save_model_monitor", type=str, help="Save model callback metrics to monitor."
    )
    train_parser.add_argument(
        "--save_model_mode",
        type=str,
        choices=["min", "max"],
        help="Save model callback metrics mode for metrics.",
    )
    train_parser.add_argument(
        "--checkpoints_dir",
        type=str,
        metavar="[path]",
        help="Path to checkpoints directiory.",
    )
    train_parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of subprocesses for data loading. Use 0 for single-threaded loading.",
    )
    train_parser.add_argument(
        "--pin_memory", type=bool, help="Enable pinned memory for faster GPU transfer."
    )
    train_parser.add_argument(
        "--do_reduce_labels",
        type=bool,
        help="Reduce all labels by 1, converting 0 to 255.",
    )
    # Add alpha loss arguments
    train_parser.add_argument(
        "--alpha", type=float, help="Cross-Entropy loss alpha weight."
    )
    train_parser.add_argument(
        "--ignore_index", type=int, help="loss function ignore index."
    )
    train_parser.add_argument(
        "--weighting_strategy",
        type=str,
        # Unified parameter options
        choices=["none", "balanced", "max", "sum", "raw"],
        metavar="[none | balanced | max | sum | raw]",
        help="Strategy for computing class weights: "
        "'raw' for inverse frequency without normalization, "
        "'balanced' to normalize weights so their sum equals 1, "
        "'max' to normalize weights so the max weight equals 1, "
        "'sum' to normalize weights so their sum equals 1, "
        "or 'none' to disable class weights.",
    )
    return train_parser


def create_evaluate_parser(subparsers, common_parser):
    """
    Create the evaluate subparser.
    """
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the model.",
        parents=[common_parser],
        formatter_class=RawTextHelpFormatter,
    )
    evaluate_parser.add_argument(
        "-v", "--version", type=int, help="Model version from checkpoints to evaluate."
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
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config yaml file. (default: config.yaml)",
        metavar="[path]",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command", required=False, title="Commands", metavar="[command]"
    )

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
    from ucs.utils.config import Config

    args_dict, config_path, command = handle_arparse(args)

    # Load configuration
    config = Config.load_config(config_path, **args_dict)

    if command == "train":
        from ucs.model.train import train

        train(config, resume_version=args.resume)
    elif command == "evaluate":
        from ucs.model.evaluate import evaluate

        evaluate(config, version=args.version)


if __name__ == "__main__":
    main()
