"""
Training loop for SegFormer-B5 U-Net on Cityscapes.
"""

import os
from argparse import ArgumentParser
import random

from torchvision.transforms import v2
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype, InterpolationMode,
)
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassF1Score
import torchvision.transforms.functional as TF
from model import Model
import torch.nn.functional as F


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


# Mapping train IDs to color
train_id_to_color = {
    cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255
}
train_id_to_color[255] = (0, 0, 0)


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for SegFormer-B5 U-Net")
    parser.add_argument("--data-dir",          type=str,   default="./data/cityscapes")
    parser.add_argument("--batch-size",         type=int,   default=4)
    parser.add_argument("--epochs",             type=int,   default=80)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--num-workers",        type=int,   default=10)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--experiment-id",      type=str,   default="segformer-unet")
    parser.add_argument("--dino-fine-tune",     type=bool,  default=False)
    parser.add_argument("--ce-weight",          type=float, default=2.0)
    parser.add_argument("--dice-weight",        type=float, default=0.5)
    parser.add_argument("--focal-weight",       type=float, default=2.0)
    parser.add_argument("--accumulation-steps", type=int,   default=4)
    return parser


def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = torch.tensor([
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
    ]).to(device)

    img_transform = Compose([
        ToImage(),
        Resize((512, 1024)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    target_transform = Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine", target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )
    valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine", target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=4,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=2, shuffle=False,
        num_workers=args.num_workers,
    )

    model = Model(in_channels=3, n_classes=19, dino_fine_tune=args.dino_fine_tune).to(device)

    # Loss functions
    criterion       = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    dice_criterion  = smp.losses.DiceLoss(mode="multiclass", classes=19, ignore_index=255)
    focal_criterion = smp.losses.FocalLoss(mode="multiclass", ignore_index=255)

    # Metrics
    dice_metric   = MulticlassF1Score(num_classes=19, average="macro", ignore_index=255).to(device)
    server_metric = MulticlassF1Score(num_classes=19, average=None,    ignore_index=255).to(device)

    # Separate encoder and decoder param groups
    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params    = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": args.lr * 0.01},  # tiny lr for pretrained encoder
            {"params": head_params,    "lr": args.lr},          # normal lr for decoder
        ],
        weight_decay=1e-4,
    )

    # Poly LR standard for segmentation
    total_steps = args.epochs * len(train_dataloader) // args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max((1 - step / total_steps) ** 0.9, 1e-6),
    )

    # Augmentation transforms (defined once outside loop)
    color_jitter = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1).to(device)
    blur         = v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)).to(device)

    best_valid_loss       = float("inf")
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")
 
        model.train()
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            # Data augmentation
            if torch.rand(1) < 0.5:
                images = torch.flip(images, dims=[3])
                labels = torch.flip(labels, dims=[2])
            if torch.rand(1) < 0.1:
                images = TF.rgb_to_grayscale(images, num_output_channels=3)
            if torch.rand(1) < 0.5:
                images = color_jitter(images)
            if torch.rand(1) < 0.3:
                images = blur(images)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                crossEntropy_loss = criterion(outputs, labels)
                dice_loss         = dice_criterion(outputs, labels)
                focal_loss        = focal_criterion(outputs, labels)
                loss = (
                    args.ce_weight    * crossEntropy_loss
                    + args.dice_weight  * dice_loss
                    + args.focal_weight * focal_loss
                )

            (loss / args.accumulation_steps).backward()

            if (i + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                wandb.log(
                    {
                        "train_loss":          loss.item(),
                        "cross_entropy_loss":  args.ce_weight    * crossEntropy_loss.item(),
                        "dice_loss":           args.dice_weight  * dice_loss.item(),
                        "focal_loss":          args.focal_weight * focal_loss.item(),
                        "learning_rate":       optimizer.param_groups[1]["lr"],
                        "epoch":               epoch + 1,
                    },
                    step=epoch * len(train_dataloader) + i,
                )

        # Validation
        model.eval()
        with torch.no_grad():
            losses              = []
            crossEntropy_losses = []
            dice_losses         = []
            focal_losses        = []

            dice_metric.reset()
            server_metric.reset()

            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs           = model(images)
                    crossEntropy_loss = criterion(outputs, labels)
                    dice_loss         = dice_criterion(outputs, labels)
                    focal_loss        = focal_criterion(outputs, labels)
                    loss = (
                        args.ce_weight    * crossEntropy_loss
                        + args.dice_weight  * dice_loss
                    )

                crossEntropy_losses.append(crossEntropy_loss.item())
                dice_losses.append(dice_loss.item())
                focal_losses.append(focal_loss.item())
                losses.append(loss.item())

                dice_metric.update(outputs.argmax(dim=1), labels)
                server_metric.update(outputs.argmax(dim=1), labels)

                if i == 0:
                    pred_vis = outputs.softmax(1).argmax(1).unsqueeze(1)
                    lbl_vis  = labels.unsqueeze(1)
                    wandb.log(
                        {
                            "predictions": [wandb.Image(
                                make_grid(convert_train_id_to_color(pred_vis).cpu(), nrow=4)
                                .permute(1, 2, 0).numpy()
                            )],
                            "labels": [wandb.Image(
                                make_grid(convert_train_id_to_color(lbl_vis).cpu(), nrow=4)
                                .permute(1, 2, 0).numpy()
                            )],
                        },
                        step=(epoch + 1) * len(train_dataloader) - 1,
                    )

            valid_loss      = sum(losses) / len(losses)
            mean_dice_score = dice_metric.compute()
            per_class       = server_metric.compute()

            flat         = per_class[[0, 1]].mean()
            construction = per_class[[2, 3, 4]].mean()
            object_cat   = per_class[[5, 6, 7]].mean()
            nature       = per_class[[8, 9]].mean()
            sky          = per_class[[10]].mean()
            human        = per_class[[11, 12]].mean()
            vehicle      = per_class[[13, 14, 15, 16, 17, 18]].mean()
            server_mean  = torch.stack(
                [flat, construction, object_cat, nature, sky, human, vehicle]
            ).mean()

            wandb.log(
                {
                    "valid_loss":              valid_loss,
                    "valid_cross_entropy_loss": args.ce_weight   * sum(crossEntropy_losses) / len(crossEntropy_losses),
                    "valid_dice_loss":          args.dice_weight * sum(dice_losses) / len(dice_losses),
                    "valid_focal_loss":         args.focal_weight * sum(focal_losses) / len(focal_losses),
                    "valid_dice_score":         mean_dice_score.item(),
                    "server/mean_dice":         server_mean.item(),
                    "server/flat_dice":         flat.item(),
                    "server/construction_dice": construction.item(),
                    "server/object_dice":       object_cat.item(),
                    "server/nature_dice":       nature.item(),
                    "server/sky_dice":          sky.item(),
                    "server/human_dice":        human.item(),
                    "server/vehicle_dice":      vehicle.item(),
                },
                step=(epoch + 1) * len(train_dataloader) - 1,
            )

            print(f"  val_loss={valid_loss:.4f}  dice={mean_dice_score:.4f}  server={server_mean:.4f}")

            if epoch % 4 == 0:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, f"checkpoint-epoch={epoch:04}.pt"))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt",
                )
                torch.save(model.state_dict(), current_best_model_path)

        scheduler.step()

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt"),
    )
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)