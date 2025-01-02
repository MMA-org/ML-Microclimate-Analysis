import os
import argparse
from argparse import RawTextHelpFormatter
from core.train_parser import TrainParser
from core.eval_parser import EvalParser
from utils.config import Config

# Disable Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def parse_args():
    """
    Parse top-level command-line arguments and subcommands (train/evaluate).
    """
    # Parent parser for global arguments

    # Main parser
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=(
            "LandCover Model:\n"
            " This script allows you to train or evaluate a SegFormer model for land-cover semantic segmentation tasks.\n"
        ),
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="Path to config yaml file. (default: config.yaml)", metavar="[path]")

    # Subparsers for train and evaluate commands
    subparsers = parser.add_subparsers(
        dest="command", required=True, title="commands", metavar="")

    # Train command parser
    train_parser = TrainParser().get_parser()
    subparsers.add_parser(
        "train", parents=[train_parser], add_help=False,
        help="Train the SegFormer model"
    )

    # Evaluate command parser
    eval_parser = EvalParser().get_parser()
    subparsers.add_parser(
        "evaluate", parents=[eval_parser], add_help=False,
        help="Evaluate the SegFormer model"
    )

    return parser


def handle_arparse(args):
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
    parser = parse_args()
    args = parser.parse_args()
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
