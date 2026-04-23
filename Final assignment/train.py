import os
from argparse import ArgumentParser
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
)
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassF1Score
from transformers import SegformerForSemanticSegmentation
import wandb

from model import Model


id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img):
    return label_img.apply_(lambda x: id_to_trainid[x])


train_id_to_color = {
    cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255
}
train_id_to_color[255] = (0, 0, 0)


def convert_train_id_to_color(prediction):
    batch, _, h, w = prediction.shape
    color_image = torch.zeros((batch, 3, h, w), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining cross-entropy. Only backprops through the
    hardest pixels (those where the model is least confident), which helps
    significantly on small/rare classes like poles, riders, and traffic signs.
    Shrivastava et al. (2016) https://arxiv.org/abs/1604.03540
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100_000):
        super().__init__()
        self.ignore = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept

    def forward(self, pred, target):
        n, c, h, w = pred.shape
        flat_target = target.view(-1)
        valid = flat_target != self.ignore

        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        prob = F.softmax(pred.permute(0, 2, 3, 1).reshape(-1, c), dim=1)
        gt_prob = prob[valid, flat_target[valid]].detach()

        n_keep = max((gt_prob < self.thresh).sum().item(), self.min_kept)
        n_keep = min(n_keep, gt_prob.numel())
        thresh_p = gt_prob.sort()[0][n_keep - 1].item()

        hard_mask = gt_prob <= thresh_p
        per_pixel = F.cross_entropy(
            pred, target, ignore_index=self.ignore, reduction="none"
        )
        return per_pixel.view(-1)[valid][hard_mask].mean()


def logit_kd_loss(student_logits, teacher_logits, labels, T):
    valid = labels != 255
    if valid.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)
    s = student_logits.permute(0, 2, 3, 1)[valid]
    t = teacher_logits.permute(0, 2, 3, 1)[valid]
    return F.kl_div(
        F.log_softmax(s / T, dim=1),
        F.softmax(t / T, dim=1),
        reduction="batchmean",
    ) * (T**2)


def feature_kd_loss(student_feat, teacher_feat):
    """
    Pixel-wise cosine similarity between student and teacher feature maps.
    Scale-agnostic so it works regardless of activation magnitude differences.
    Inspired by CWD: Shu et al. (2021) https://arxiv.org/abs/2011.13256
    """
    sf = F.normalize(student_feat, dim=1)
    tf = F.normalize(teacher_feat.detach(), dim=1)
    return (1.0 - (sf * tf).sum(dim=1)).mean()


class FeatureAlignProjector(nn.Module):
    """
    Training-only module that projects the student bottleneck (48ch) into
    the teacher's feature space (512ch) for feature-level distillation.
    Not saved with the student zero inference overhead.
    """

    def __init__(self, student_ch=48, teacher_ch=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_ch, teacher_ch, 1, bias=False),
            nn.BatchNorm2d(teacher_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


def cutmix(images, labels, alpha=1.0):
    """
    Yun et al. (2019) https://arxiv.org/abs/1905.04899
    Swaps a random rectangular patch between two images in the batch,
    labels included. Forces the model to use global context rather than
    local texture shortcuts.
    """
    lam = (
        torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha))
        .sample()
        .item()
    )
    B, _, H, W = images.shape
    idx = torch.randperm(B, device=images.device)

    cx, cy = int(W * random.random()), int(H * random.random())
    bw, bh = int(W * (1 - lam) ** 0.5), int(H * (1 - lam) ** 0.5)
    x1, x2 = max(cx - bw // 2, 0), min(cx + bw // 2, W)
    y1, y2 = max(cy - bh // 2, 0), min(cy + bh // 2, H)

    images = images.clone()
    labels = labels.clone()
    images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    labels[:, y1:y2, x1:x2] = labels[idx, y1:y2, x1:x2]
    return images, labels


def get_args_parser():
    parser = ArgumentParser("Efficient Cityscapes — v2")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="efficient-v2")
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    # loss weights
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=1.5)
    parser.add_argument("--kd-logit-weight", type=float, default=0.5)
    parser.add_argument("--kd-feat-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=4.0)
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

    class_weights = torch.tensor(
        [
            0.8,
            1.0,
            1.0,
            2.0,
            2.0,
            2.5,
            2.5,
            2.5,
            1.0,
            1.5,
            0.8,
            3.0,
            3.5,
            1.0,
            2.5,
            2.5,
            3.0,
            3.5,
            3.0,
        ]
    ).to(device)

    img_transform = Compose(
        [
            ToImage(),
            Resize((512, 1024)),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    target_transform = Compose(
        [
            ToImage(),
            Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64),
        ]
    )

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
        batch_size=2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # output_hidden_states=True to get intermediate features for feature-level KD
    teacher = (
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            output_hidden_states=True,
        )
        .to(device)
        .eval()
    )
    for p in teacher.parameters():
        p.requires_grad = False

    model = Model(in_channels=3, n_classes=19).to(device)
    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        model.load_state_dict(
            torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
        )
        print(f"Loaded student from: {args.pretrained_ckpt}")

    # Projects student bottleneck (48ch) to teacher feature space (512ch)
    feat_projector = FeatureAlignProjector(student_ch=48, teacher_ch=512).to(device)

    ohem_criterion = OHEMLoss(ignore_index=255, thresh=0.9, min_kept=20000)
    dice_criterion = smp.losses.DiceLoss(
        mode="multiclass", classes=19, ignore_index=255
    )
    focal_criterion = smp.losses.FocalLoss(mode="multiclass", ignore_index=255)

    dice_metric = MulticlassF1Score(
        num_classes=19, average="macro", ignore_index=255
    ).to(device)
    server_metric = MulticlassF1Score(
        num_classes=19, average=None, ignore_index=255
    ).to(device)

    backbone_params = [
        p for n, p in model.named_parameters() if n.startswith("backbone.")
    ]
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("backbone.")
    ]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr * 2.5},
            {"params": feat_projector.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    # Poly LR — standard for segmentation (DeepLab, DDRNet, SegFormer all use this)
    total_steps = args.epochs * len(train_dataloader) // args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max((1 - step / total_steps) ** 0.9, 1e-6),
    )

    color_jitter = v2.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    ).to(device)
    blur = v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)).to(device)

    best_valid_loss = float("inf")
    current_best_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        model.train()
        feat_projector.train()
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            if torch.rand(1) < 0.5:
                images = torch.flip(images, dims=[3])
                labels = torch.flip(labels, dims=[2])
            if torch.rand(1) < 0.1:
                images = TF.rgb_to_grayscale(images, num_output_channels=3)
            if torch.rand(1) < 0.5:
                images = color_jitter(images)
            if torch.rand(1) < 0.3:
                images = blur(images)
            if torch.rand(1) < 0.25:
                images, labels = cutmix(images, labels)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                # student_bottleneck = model._bottleneck  # set inside Model.forward()

                # with torch.no_grad():
                # teacher_out    = teacher(pixel_values=images)
                # teacher_logits = F.interpolate(
                #     teacher_out.logits, size=images.shape[2:],
                #     mode="bilinear", align_corners=False,
                # )
                # # Resize teacher's last hidden state to match student bottleneck (H/16)
                # teacher_feat = F.interpolate(
                #     teacher_out.hidden_states[-1],
                #     size=student_bottleneck.shape[2:],
                #     mode="bilinear", align_corners=False,
                # )

                ohem_loss = ohem_criterion(outputs, labels)
                dice_loss = dice_criterion(outputs, labels)
                focal_loss = focal_criterion(outputs, labels)
                # kd_loss    = logit_kd_loss(outputs, teacher_logits, labels, args.temperature)
                # feat_loss  = feature_kd_loss(feat_projector(student_bottleneck.float()),
                #                              teacher_feat.float())

                loss = (
                    args.ce_weight * ohem_loss
                    + args.dice_weight * dice_loss
                    + args.focal_weight * focal_loss
                    # + args.kd_logit_weight * kd_loss
                    # + args.kd_feat_weight  * feat_loss
                )

            (loss / args.accumulation_steps).backward()

            if (i + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(feat_projector.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/ohem": args.ce_weight * ohem_loss.item(),
                        "train/dice": args.dice_weight * dice_loss.item(),
                        "train/focal": args.focal_weight * focal_loss.item(),
                        # "train/kd_logit":      args.kd_logit_weight * kd_loss.item(),
                        # "train/kd_feat":       args.kd_feat_weight * feat_loss.item(),
                        "train/lr": optimizer.param_groups[1]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=epoch * len(train_dataloader) + i,
                )

        model.eval()
        feat_projector.eval()
        with torch.no_grad():
            losses = []
            dice_metric.reset()
            server_metric.reset()

            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(images)
                    ohem_loss = ohem_criterion(outputs, labels)
                    dice_loss = dice_criterion(outputs, labels)
                    val_loss = args.ce_weight * ohem_loss + args.dice_weight * dice_loss

                losses.append(val_loss.item())
                dice_metric.update(outputs.argmax(dim=1), labels)
                server_metric.update(outputs.argmax(dim=1), labels)

                if i == 0:
                    pred_vis = outputs.softmax(1).argmax(1).unsqueeze(1)
                    lbl_vis = labels.unsqueeze(1)
                    wandb.log(
                        {
                            "vis/predictions": [
                                wandb.Image(
                                    make_grid(
                                        convert_train_id_to_color(pred_vis).cpu(),
                                        nrow=4,
                                    )
                                    .permute(1, 2, 0)
                                    .numpy()
                                )
                            ],
                            "vis/labels": [
                                wandb.Image(
                                    make_grid(
                                        convert_train_id_to_color(lbl_vis).cpu(), nrow=4
                                    )
                                    .permute(1, 2, 0)
                                    .numpy()
                                )
                            ],
                        },
                        step=(epoch + 1) * len(train_dataloader) - 1,
                    )

            valid_loss = sum(losses) / len(losses)
            mean_dice_score = dice_metric.compute()
            per_class = server_metric.compute()

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
                    "val/loss": valid_loss,
                    "val/dice_score": mean_dice_score.item(),
                    "server/mean_dice": server_mean.item(),
                    "server/flat": flat.item(),
                    "server/construction": construction.item(),
                    "server/object": object_cat.item(),
                    "server/nature": nature.item(),
                    "server/sky": sky.item(),
                    "server/human": human.item(),
                    "server/vehicle": vehicle.item(),
                },
                step=(epoch + 1) * len(train_dataloader) - 1,
            )

            print(
                f"  val_loss={valid_loss:.4f}  dice={mean_dice_score:.4f}  server={server_mean:.4f}"
            )

            if epoch % 4 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"checkpoint-epoch={epoch:04}.pt"),
                )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_path and os.path.exists(current_best_path):
                    os.remove(current_best_path)
                current_best_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt",
                )
                torch.save(model.state_dict(), current_best_path)

    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt"
        ),
    )
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
