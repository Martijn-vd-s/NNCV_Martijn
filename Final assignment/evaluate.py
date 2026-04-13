import os
import json
import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Normalize, ToImage, ToDtype
from model import Model
import torch.nn.functional as F


CATEGORY_MAPPING = {
    "Flat":         [0, 1],
    "Construction": [2, 3, 4],
    "Object":       [5, 6, 7],
    "Nature":       [8, 9],
    "Sky":          [10],
    "Human":        [11, 12],
    "Vehicle":      [13, 14, 15, 16, 17, 18],
}


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return torch.bincount(n * a[k] + b[k], minlength=n**2).reshape(n, n)


def compute_efficiency_metrics(model, device, input_size=(1, 3, 1024, 2048)):
    """Compute GFLOPs, TFLOPs, FPS and model size MB"""
    dummy = torch.randn(*input_size).to(device)

    # GFLOPs via fvcore (same tool the CodaLab server uses)
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy)
            flops.unsupported_ops_warnings(False)
            gflops = flops.total() / 1e9
    except ImportError:
        print("WARNING: fvcore not installed, skipping GFLOPs count")
        print("  install with: pip install fvcore")
        gflops = None

    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        for _ in range(30):
            model(dummy)
        torch.cuda.synchronize() if device.type == "cuda" else None
        fps = 30 / (time.time() - t0)

    # Model size in MB (parameters + buffers)
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_bytes + buffer_bytes) / 1e6

    return gflops, fps, size_mb



def sliding_window_inference(
    model, image_tensor, window_size=(512, 1024), stride_rate=0.5
    ):
    device = image_tensor.device
    B, _, H, W = image_tensor.shape
    w_h, w_w = window_size

    stride_h = int(w_h * stride_rate)
    stride_w = int(w_w * stride_rate)

    num_classes = 19
    preds = torch.zeros((B, num_classes, H, W), device=device)
    count_map = torch.zeros((B, 1, H, W), device=device)

    h_starts = list(range(0, max(H - w_h + stride_h, 1), stride_h))
    w_starts = list(range(0, max(W - w_w + stride_w, 1), stride_w))

    for y in h_starts:
        for x in w_starts:
            y1 = min(y, H - w_h)
            y2 = y1 + w_h
            x1 = min(x, W - w_w)
            x2 = x1 + w_w

            crop = image_tensor[:, :, y1:y2, x1:x2]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs_normal = model(crop)

                crop_flipped = torch.flip(crop, dims=[3])
                outputs_flipped = model(crop_flipped)
                outputs_flipped = torch.flip(outputs_flipped, dims=[3])

            crop_pred = (outputs_normal.float() + outputs_flipped.float()) / 2.0
            crop_probs = torch.nn.functional.softmax(crop_pred, dim=1)

            preds[:, :, y1:y2, x1:x2] += crop_probs
            count_map[:, :, y1:y2, x1:x2] += 1

    final_preds = preds / count_map
    return final_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    img_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    target_transform = Compose([
        ToImage(),
        ToDtype(torch.int64),
    ])

    val_dataset = Cityscapes(
        root="./data/cityscapes", split="val", mode="fine",
        target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10)

    model = Model(in_channels=3, n_classes=19, dino_fine_tune=False).to(device)

    checkpoint_path = "checkpoints/eff + unet-training V1.1/checkpoint-epoch=0008.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # compute efficiency metrics first (server uses 1024x2048 for FLOP counting)
    print("\nComputing efficiency metrics...")
    gflops, fps, size_mb = compute_efficiency_metrics(model, device, input_size=(1, 3, 1024, 2048))
    tflops = gflops / 1000 if gflops else None

    print(f"  Model size : {size_mb:.2f} MB")
    print(f"  GFLOPs     : {gflops:.2f}" if gflops else "  GFLOPs     : N/A")
    print(f"  FPS        : {fps:.2f}")

    # run validation
    num_classes = 19
    hist = torch.zeros((num_classes, num_classes), device=device)
    id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

    print("\nRunning inference on full validation set...")
    with torch.no_grad():
        from tqdm import tqdm
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.apply_(lambda x: id_to_trainid.get(x, 255)).long().squeeze(1).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                # outputs = model(images)
                preds = []
                for scale in [1.0]:
                    if scale != 1.0:
                        h = (
                            round(images.shape[2] * scale / 16) * 16
                        )  # keep divisible by 16
                        w = round(images.shape[3] * scale / 16) * 16
                        scaled = F.interpolate(
                            images, size=(h, w), mode="bilinear", align_corners=False
                        )
                    else:
                        scaled = images

                    pred_scale = sliding_window_inference(
                        model=model,
                        image_tensor=scaled,
                        window_size=(512, 1024),
                        stride_rate=1,
                    )

                    pred_scale = F.interpolate(
                        pred_scale,
                        size=images.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    preds.append(pred_scale)

                outputs = torch.stack(preds).mean(dim=0)

            predictions = outputs.argmax(dim=1)
            hist += fast_hist(labels.flatten(), predictions.flatten(), num_classes)

    # compute per-category Dice and IoU
    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp

    results = {}
    total_iou  = 0.0
    total_dice = 0.0

    for cat_name, class_ids in CATEGORY_MAPPING.items():
        cat_tp = tp[class_ids].sum().item()
        cat_fp = fp[class_ids].sum().item()
        cat_fn = fn[class_ids].sum().item()

        iou  = cat_tp / (cat_tp + cat_fp + cat_fn) if (cat_tp + cat_fp + cat_fn) > 0 else 0.0
        dice = (2 * cat_tp) / (2 * cat_tp + cat_fp + cat_fn) if (2 * cat_tp + cat_fp + cat_fn) > 0 else 0.0

        results[f"Dice_{cat_name}"] = round(dice, 4)
        results[f"IoU_{cat_name}"]  = round(iou,  4)
        total_iou  += iou
        total_dice += dice

    results["MeanDice"]   = round(total_dice / len(CATEGORY_MAPPING), 4)
    results["MeanIoU"]    = round(total_iou  / len(CATEGORY_MAPPING), 4)
    results["NumSamples"] = len(val_dataset)

    # efficiency score
    if tflops:
        results["GFLOPs"]     = round(gflops, 4)
        results["TFLOPs"]     = round(tflops, 6)
        results["FPS"]        = round(fps, 2)
        results["ModelSizeMB"]= round(size_mb, 2)
        results["Efficiency_Dice_per_TFLOPs"] = round(results["MeanDice"] / tflops, 4)

    print("\n--- Final Metrics ---")
    print(json.dumps(results, indent=2))

    with open("final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to final_metrics.json")


if __name__ == "__main__":
    main()