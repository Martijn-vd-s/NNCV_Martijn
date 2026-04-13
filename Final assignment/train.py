import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose, Normalize, ToImage, ToDtype, Resize,
    RandomHorizontalFlip, ColorJitter, InterpolationMode,
)
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
from model import Model


CITYSCAPES_ROOT = "./data/cityscapes"
CHECKPOINT_DIR  = "./checkpoints/distillation"
PRETRAINED_CKPT = "checkpoints/eff + unet-training V1.1/best_model-epoch=0051-val_loss=0.5333827495574951.pt"
EPOCHS          = 60
BATCH_SIZE      = 4
LR              = 5e-4
TEMPERATURE     = 4.0
ALPHA           = 0.5

CATEGORY_MAPPING = {
    "Flat":         [0, 1],
    "Construction": [2, 3, 4],
    "Object":       [5, 6, 7],
    "Nature":       [8, 9],
    "Sky":          [10],
    "Human":        [11, 12],
    "Vehicle":      [13, 14, 15, 16, 17, 18],
}

class_weights = torch.tensor([
    0.5, 0.8, 0.8, 2.0, 2.0, 3.5, 3.5, 3.5,
    0.8, 1.2, 0.5, 4.0, 4.5, 0.8, 3.0, 3.5, 4.0, 4.5, 3.5,
])


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return torch.bincount(n * a[k] + b[k], minlength=n**2).reshape(n, n)


def compute_dice(hist):
    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp
    total = 0.0
    for ids in CATEGORY_MAPPING.values():
        cat_tp = tp[ids].sum().item()
        cat_fp = fp[ids].sum().item()
        cat_fn = fn[ids].sum().item()
        denom  = 2 * cat_tp + cat_fp + cat_fn
        total += (2 * cat_tp / denom) if denom > 0 else 0.0
    return total / len(CATEGORY_MAPPING)


def distillation_loss(student_logits, teacher_logits, labels, weights):
    ce_loss = F.cross_entropy(student_logits, labels, weight=weights, ignore_index=255)

    valid   = labels != 255
    if valid.sum() == 0:
        return ce_loss

    s = student_logits.permute(0, 2, 3, 1)[valid]
    t = teacher_logits.permute(0, 2, 3, 1)[valid]

    kd_loss = F.kl_div(
        F.log_softmax(s / TEMPERATURE, dim=1),
        F.softmax(t / TEMPERATURE, dim=1),
        reduction="batchmean",
    ) * (TEMPERATURE ** 2)

    return ALPHA * ce_loss + (1 - ALPHA) * kd_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

    train_img_transform = Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_img_transform = Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    target_transform = Compose([
        ToImage(),
        Resize((512, 1024), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    train_loader = DataLoader(
        Cityscapes(CITYSCAPES_ROOT, split="train", mode="fine", target_type="semantic",
                   transform=train_img_transform, target_transform=target_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        Cityscapes(CITYSCAPES_ROOT, split="val", mode="fine", target_type="semantic",
                   transform=val_img_transform, target_transform=target_transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
    )

    teacher = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = Model(in_channels=3, n_classes=19, dino_fine_tune=False).to(device)
    student.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=device, weights_only=True))

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()
    weights   = class_weights.to(device)

    best_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            images = images.to(device)
            labels = labels.apply_(lambda x: id_to_trainid.get(x, 255)).long().squeeze(1).to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                student_logits = student(images)

                with torch.no_grad():
                    teacher_out    = teacher(pixel_values=images)
                    teacher_logits = F.interpolate(
                        teacher_out.logits, size=images.shape[2:],
                        mode="bilinear", align_corners=False,
                    )

                loss = distillation_loss(student_logits, teacher_logits, labels, weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        print(f"  loss: {total_loss / len(train_loader):.4f}  lr: {scheduler.get_last_lr()[0]:.6f}")

        student.eval()
        hist = torch.zeros((19, 19), device=device)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                images = images.to(device)
                labels = labels.apply_(lambda x: id_to_trainid.get(x, 255)).long().squeeze(1).to(device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = student(images)

                predictions = outputs.argmax(dim=1)
                hist += fast_hist(labels.flatten(), predictions.flatten(), 19)

        mean_dice = compute_dice(hist)
        print(f"  val dice: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model-epoch={epoch:04d}-dice={mean_dice:.4f}.pt")
            torch.save(student.state_dict(), ckpt_path)
            print(f"  saved: {ckpt_path}")

        torch.save(student.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))

    print(f"\nbest dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()