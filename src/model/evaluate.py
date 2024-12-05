import argparse
from model.lightning_model import SegformerFinetuner
from src.data.loader import Loader
from utils import load_config, lc_id2label
from pytorch_lightning import Trainer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer")
    parser.add_argument(
        "--config", type=str, default="default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str,
                        required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    # load config
    config = load_config(args.config)

    checkpoint = Path(config["project"]["checkpoint_dir"], args.checkpoint)

    # Prepare test dataloader
    loader = Loader(config)
    test_loader = loader.get_dataloader("test")

    # Load model from checkpoint
    model = SegformerFinetuner.load_from_checkpoint(
        checkpoint, lc_id2label)

    # Evaluate
    trainer = Trainer()
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
