import argparse
import os
from argparse import RawTextHelpFormatter

# Disable Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(
        usage="lcm [command] [--help|-h] [options]",
        formatter_class=RawTextHelpFormatter,
        description=(
            "LandCover Model:\n"
            " This script allows you to train or evaluate a SegFormer model for land-cover semantic segmentation tasks.\n"
        ))

    subparsers = parser.add_subparsers(dest="command", required=True, title="commands",
                                       metavar="")

    # Define the 'train' command
    train_parser = subparsers.add_parser(
        "train", formatter_class=argparse.RawTextHelpFormatter, help="Train the SegFormer model", usage="mc train [--help|-h] [options]"
    )

    train_parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="Path to config yaml file. (default: config.yaml)", metavar=""
    )

    train_parser.add_argument(
        "-r", "--resume", type=str,
        help=(
            "Specify the model version number to resume training from.\n"
            "For example '3' will look for checkpoints in the directory corresponding to version_3.\n"
            "Ensure that a valid checkpoint exists in the specified version directory."
        ), metavar=""
    )

    # Define the 'eval' command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate the SegFormer model", usage="mc eval [--help|-h] [options]"
    )
    eval_parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", help="Path to config yaml file. (default: config.yaml)", metavar=""
    )
    eval_parser.add_argument(
        "-v", "--version", type=str, default="0", help="Model version to evaluate. Default is '0'. Specify the version folder if using a specific checkpoint.", metavar=""
    )

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    from utils import Config
    config = Config(args.config)

    if args.command == "train":
        from model.train import train
        train(config,
              resume_version=args.resume)
    elif args.command == "evaluate":
        from model.evaluate import evaluate
        evaluate(config, version=args.version)


if __name__ == "__main__":
    main()
