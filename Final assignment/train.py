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

from html import parser
import os
from argparse import ArgumentParser
import random

# from cv2 import blur
from torchvision.transforms import v2
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
import torchvision.transforms.functional as TF
from model import Model
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex
from predict import sliding_window_inference


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
    parser.add_argument(
        "--ce-weight", type=float, default=2.0, help="Weight for Cross Entropy Loss"
    )
    parser.add_argument(
        "--dice-weight", type=float, default=0.5, help="Weight for Dice Loss"
    )
    parser.add_argument(
        "--focal-weight", type=float, default=2.0, help="Weight for Focal Loss"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (because of the small batch size with full sized images, we need to accumulate gradients over multiple batches to effectively have a larger batch size)",
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

    class_weights = torch.tensor(
        [
            0.8,  # road
            1.0,  # sidewalk
            1.0,  # building
            2.0,  # wall
            2.0,  # fence
            2.5,  # pole
            2.5,  # traffic light
            2.5,  # traffic sign
            1.0,  # vegetation
            1.5,  # terrain
            0.8,  # sky
            3.0,  # person
            3.5,  # rider
            1.0,  # car
            2.5,  # truck
            2.5,  # bus
            3.0,  # train
            3.5,  # motorcycle
            3.0,  # bicycle
        ]
    ).to(device)

    # Define the transforms to apply to the data
    img_transform = Compose(
        [
            ToImage(),
            Resize(
                (512, 1024)
            ),  # increase the resolution to 512x1024, since the DINO model is pretrained on higher resolution images, this should help with the performance of the model
            ToDtype(torch.float32, scale=True),
            Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # normalization values from ImageNet, since the DINO model is pretrained on ImageNet
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
        prefetch_factor=4,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=2,  # use smaller batch size with full sized images to avoid out of memory errors during validation
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
        dino_fine_tune=args.dino_fine_tune,  # Whether to fine-tune the DINO model
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255
    )  # Ignore the void class
    dice_criterion = smp.losses.DiceLoss(
        mode="multiclass", classes=19, ignore_index=255
    )  # Dice loss for multi-class segmentation
    focal_criterion = smp.losses.FocalLoss(mode="multiclass", ignore_index=255)

    # Dice metric for evaluation (not used in training, but can be logged during validation)
    dice_metric = MulticlassF1Score(
        num_classes=19, average="macro", ignore_index=255
    ).to(device)

    server_metric = MulticlassF1Score(
        num_classes=19,
        average=None,  # returns a score per class, not averaged
        ignore_index=255,
    ).to(device)

    backbone_params = [
        param for name, param in model.named_parameters()
        if name.startswith(("enc1", "enc2", "enc3", "enc4", "enc5"))
    ]
    head_params = [
        param for name, param in model.named_parameters()
        if not name.startswith(("enc1", "enc2", "enc3", "enc4", "enc5"))
    ]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1}, 
            {"params": head_params,     "lr": args.lr},       
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

        # define the color jitter transform outside the loop to avoid creating a new instance for each batch
        color_jitter = v2.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ).to(device)

        blur = v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)).to(device)
        random_crop = v2.RandomCrop(size=(640, 1280)).to(device)

        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            # # randomly scale images and the labels
            # scale = random.uniform(0.75, 1.5)
            # new_h, new_w = int(1024 * scale), int(2048 * scale)
            # images = F.interpolate(
            #     images, size=(new_h, new_w), mode="bilinear", align_corners=False
            # )
            # labels = F.interpolate(
            #     labels.float(), size=(new_h, new_w), mode="nearest"
            # ).long()

            # # randomly crop images and labels
            # crop_i, crop_j, crop_h, crop_w = v2.RandomCrop.get_params(
            #     images, output_size=(512, 1024)
            # )
            # images = TF.crop(images, crop_i, crop_j, crop_h, crop_w).contiguous()
            # labels = TF.crop(labels, crop_i, crop_j, crop_h, crop_w).contiguous()

            labels = labels.long().squeeze(1)  # Remove channel dimension

            ### Data Augmentation
            # Random Horizontal Flip, only 50% of the time
            if torch.rand(1) < 0.5:
                images = torch.flip(images, dims=[3])  # Flip width dimension of image
                labels = torch.flip(
                    labels, dims=[2]
                )  # Flip width dimension of label (no channel dim)

            if torch.rand(1) < 0.1:
                images = TF.rgb_to_grayscale(
                    images, num_output_channels=3
                )  # make gray so it seems like it is dark

            # Random Color Jitter, only 50% of the time
            if torch.rand(1) < 0.5:
                images = color_jitter(images)

            if torch.rand(1) < 0.3:
                images = blur(images)
            ### End of Data Augmentation

            optimizer.zero_grad()

            # Use mixed precision for faster training and reduced memory usage
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)

                # Compute the combined loss (cross-entropy + dice loss)
                crossEntropy_loss = criterion(outputs, labels)
                dice_loss = dice_criterion(outputs, labels)
                focal_loss = focal_criterion(outputs, labels)

                # Combine the losses
                loss = (
                    (args.ce_weight * crossEntropy_loss)
                    + (args.dice_weight * dice_loss)
                    + (args.focal_weight * focal_loss)
                )

            loss = (
                loss / args.accumulation_steps
            )  # Normalize the loss by the accumulation steps
            loss.backward()

            # Gradient clipping to prevent exploding gradients, especially
            if (
                i + 1
            ) % args.accumulation_steps == 0:  # Update weights every accumulation steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "cross_entropy_loss": args.ce_weight * crossEntropy_loss.item(),
                        "dice_loss": args.dice_weight * dice_loss.item(),
                        "focal_loss": args.focal_weight * focal_loss.item(),
                        "learning_rate": optimizer.param_groups[1]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=epoch * len(train_dataloader) + i,
                )

        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            crossEntropy_losses = []
            dice_losses = []
            focal_losses = []

            # Reset the dice metric at the start of validation
            dice_metric.reset()
            server_metric.reset()

            for i, (images, labels) in enumerate(valid_dataloader):
                if (
                    i >= 5
                ):  # only validate on a subset of the validation set to save time, since we are logging the metrics to wandb, we can see the trend even with a subset of the validation set
                    break

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                # Use mixed precision for faster validating and reduced memory usage
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(images)

                    # to get small objects to appear bigger scale the images.
                    # preds = []
                    # for scale in [1.0, 1.25, 1.5]:
                    #     if scale != 1.0:
                    #         h = (
                    #             round(images.shape[2] * scale / 16) * 16
                    #         )  # keep divisible by 16
                    #         w = round(images.shape[3] * scale / 16) * 16
                    #         scaled = F.interpolate(
                    #             images,
                    #             size=(h, w),
                    #             mode="bilinear",
                    #             align_corners=False,
                    #         )
                    #     else:
                    #         scaled = images

                    #     pred_scale = sliding_window_inference(
                    #         model=model,
                    #         image_tensor=scaled,
                    #         window_size=(640, 1280),
                    #         stride_rate=0.25,
                    #     )

                    #     pred_scale = F.interpolate(
                    #         pred_scale,
                    #         size=images.shape[2:],
                    #         mode="bilinear",
                    #         align_corners=False,
                    #     )
                    #     preds.append(pred_scale)

                    # outputs = torch.stack(preds).mean(dim=0)

                    # Compute the combined loss (cross-entropy + dice loss)
                    crossEntropy_loss = criterion(outputs, labels)
                    dice_loss = dice_criterion(outputs, labels)
                    focal_loss = focal_criterion(outputs, labels)

                    # Coombine the losses
                    loss = (args.ce_weight * crossEntropy_loss) + (
                        args.dice_weight * dice_loss
                    )

                crossEntropy_losses.append(crossEntropy_loss.item())
                dice_losses.append(dice_loss.item())
                focal_losses.append(focal_loss.item())
                losses.append(loss.item())

                # Update the dice metric with the current batch's predictions and labels
                predictions = outputs.argmax(dim=1)
                dice_metric.update(predictions, labels)

                server_metric.update(predictions, labels)

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

            per_class = server_metric.compute()  # shape (19,)

            flat = per_class[[0, 1]].mean()
            construction = per_class[[2, 3, 4]].mean()
            object_cat = per_class[[5, 6, 7]].mean()
            nature = per_class[[8, 9]].mean()
            sky = per_class[[10]].mean()
            human = per_class[[11, 12]].mean()
            vehicle = per_class[[13, 14, 15, 16, 17, 18]].mean()

            server_mean = torch.stack(
                [flat, construction, object_cat, nature, sky, human, vehicle]
            ).mean()

            wandb.log(
                {
                    "valid_loss": valid_loss,
                    "valid_cross_entropy_loss": args.ce_weight
                    * sum(crossEntropy_losses)
                    / len(crossEntropy_losses),
                    "valid_dice_loss": args.dice_weight
                    * sum(dice_losses)
                    / len(dice_losses),
                    "valid_focal_loss": args.focal_weight
                    * sum(focal_losses)
                    / len(focal_losses),
                    "valid_dice_score": mean_dice_score,
                    "server/mean_dice": server_mean.item(),
                    "server/flat_dice": flat.item(),
                    "server/construction_dice": construction.item(),
                    "server/object_dice": object_cat.item(),
                    "server/nature_dice": nature.item(),
                    "server/sky_dice": sky.item(),
                    "server/human_dice": human.item(),
                    "server/vehicle_dice": vehicle.item(),
                },
                step=(epoch + 1) * len(train_dataloader) - 1,
            )

            if epoch % 4 == 0:
                periodic_path = os.path.join(
                    output_dir, f"checkpoint-epoch={epoch:04}.pt"
                )
                torch.save(model.state_dict(), periodic_path)

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
