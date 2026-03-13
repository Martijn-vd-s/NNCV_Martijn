"""
This script implements a training loop for the model. It is designed to be flexible,
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console.
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block.
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
# https://huggingface.co/docs/transformers/en/model_doc/dinov3 inspiration

import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
)
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassF1Score

from model import Model

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


# Mapping train IDs to color
train_id_to_color = {
    cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255
}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/cityscapes",
        help="Path to the training data",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--num-workers", type=int, default=10, help="Number of workers for data loaders"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="unet-training",
        help="Experiment ID for Weights & Biases",
    )
    parser.add_argument(
        "--dino-fine-tune",
        type=bool,
        default=False,
        help="Whether to fine-tune the DINO model",
    )

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random),
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    img_transform = Compose(
        [
            ToImage(),
            Resize((512, 1024)), # increase the resolution to 512x1024, since the DINO model is pretrained on higher resolution images, this should help with the performance of the model
            ToDtype(torch.float32, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # normalization values from ImageNet, since the DINO model is pretrained on ImageNet
        ]
    )

    # Target transform (mask)
    target_transform = Compose(
        [
            ToImage(),
            Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64),  # no scaling
        ]
    )

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
        dino_fine_tune=args.dino_fine_tune,  # Whether to fine-tune the DINO model
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    dice_criterion = smp.losses.DiceLoss(
        mode="multiclass", classes=19, ignore_index=255
    )  # Dice loss for multi-class segmentation

    # Dice metric for evaluation (not used in training, but can be logged during validation)
    dice_metric = MulticlassF1Score(
        num_classes=19, average="macro", ignore_index=255
    ).to(device)

    # seperate dino parameters and unet parameters for training, this way we can tune the dino model with a lower learing rate and keep the unet with a higher learning rate
    dino_params = [param for name, param in model.named_parameters() if "dino" in name]
    unet_params = [
        param for name, param in model.named_parameters() if "dino" not in name
    ]

    # Define the optimizer -- add weight dacay for regularization
    optimizer = AdamW(
        [
            {
                "params": dino_params,
                "lr": args.lr * 0.01,
            },  # tiny learning rate for dino
            {
                "params": unet_params,
                "lr": args.lr,
            },  # normal learning rate for everything else
        ],
        weight_decay=1e-4,
    )
    # Learning rate scheduler -- maybe later we can compare with other schedulers like ReduceLROnPlateau or OneCycleLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_valid_loss = float("inf")
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)

            # Compute the combined loss (cross-entropy + dice loss)
            crossEntropy_loss = criterion(outputs, labels)
            dice_loss = dice_criterion(outputs, labels)

            # Coombine the losses
            loss = crossEntropy_loss + dice_loss

            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                },
                step=epoch * len(train_dataloader) + i,
            )

        # Validation
        model.eval()
        with torch.no_grad():
            losses = []

            # Reset the dice metric at the start of validation
            dice_metric.reset()

            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                # Compute the combined loss (cross-entropy + dice loss)
                crossEntropy_loss = criterion(outputs, labels)
                dice_loss = dice_criterion(outputs, labels)

                # Coombine the losses
                loss = crossEntropy_loss + dice_loss

                losses.append(loss.item())

                # Update the dice metric with the current batch's predictions and labels
                predictions = outputs.argmax(dim=1)
                dice_metric.update(predictions, labels)

                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log(
                        {
                            "predictions": [wandb.Image(predictions_img)],
                            "labels": [wandb.Image(labels_img)],
                        },
                        step=(epoch + 1) * len(train_dataloader) - 1,
                    )

            valid_loss = sum(losses) / len(losses)
            mean_dice_score = dice_metric.compute()

            wandb.log(
                {
                    "valid_loss": valid_loss,
                    "valid_dice_score": mean_dice_score,
                },
                step=(epoch + 1) * len(train_dataloader) - 1,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt",
                )
                torch.save(model.state_dict(), current_best_model_path)

        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()

    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        ),
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
