import argparse
from ml_microclimate.DataHandler import DatasetHandler
from ml_microclimate.SegformerFinetuner import SegformerFinetuner
from ml_microclimate.TrainingHandler import TrainingHandler
from ml_microclimate.utils import lc_classes, building_class, reduce_labels, map_binary_dataset


def main(args):
    # Define dataset-specific variables based on the dataset choice
    if args.dataset_name == "buildings":
        dataset_repo = "nave1616/building-urban-climate"
        labels = building_class  # Use the building labels
        transform_function = map_binary_dataset  # Apply binary mapping for buildings
    elif args.dataset_name == "landcover":
        dataset_repo = "nave1616/landcover-urban-climate"
        labels = lc_classes  # Use the landcover labels
        transform_function = reduce_labels  # Apply label reduction for landcover
    else:
        raise ValueError(
            "Invalid dataset option. Choose either 'buildings' or 'landcover'.")

    # Initialize Dataset Handler
    dataset_handler = DatasetHandler(
        dataset_name=dataset_repo,
        model_name=args.model_name,
        transform=True
    )

    # Get the dataloaders without transformations
    train_dataloader, validation_dataloader, test_dataloader = dataset_handler.get_dataloaders(
        batch_size=args.batch_size)

    # Apply dataset-specific transformation to the dataset before training
    train_dataloader.dataset.data = transform_function(
        train_dataloader.dataset.data)
    validation_dataloader.dataset.data = transform_function(
        validation_dataloader.dataset.data)
    test_dataloader.dataset.data = transform_function(
        test_dataloader.dataset.data)

    # Initialize Segformer Finetuner
    segformer_finetuner = SegformerFinetuner(
        id2label={cls.label: cls.name for cls in labels},
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        test_dataloader=test_dataloader
    )

    # Initialize Training Handler
    training_handler = TrainingHandler(
        model=segformer_finetuner,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader,
        test_dataloader=test_dataloader,
        max_epochs=args.max_epochs
    )

    # Start Training
    training_handler.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SegFormer model for semantic segmentation.")
    parser.add_argument('--dataset_name', type=str, choices=["buildings", "landcover"],
                        required=True, help='Dataset option: "buildings" or "landcover".')
    parser.add_argument('--model_name', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512',
                        help='Pre-trained model to use for fine-tuning.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training. Default is 8.')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs to train. Default is 50.')

    args = parser.parse_args()
    main(args)
