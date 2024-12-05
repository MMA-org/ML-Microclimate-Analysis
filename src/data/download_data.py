from datasets import load_dataset
from utils import Config
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = Config(args.config)
    dataset_repo = config.dataset_path
    load_dataset(dataset_repo)


if __name__ == "__main__":
    main()
