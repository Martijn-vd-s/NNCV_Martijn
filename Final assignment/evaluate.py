import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Normalize, ToImage, ToDtype
from model import Model

CATEGORY_MAPPING = {
    "Flat": [0, 1],
    "Construction": [2, 3, 4],
    "Object": [5, 6, 7],
    "Nature": [8, 9],
    "Sky": [10],
    "Human": [11, 12],
    "Vehicle": [13, 14, 15, 16, 17, 18],
}

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return torch.bincount(n * a[k] + b[k], minlength=n**2).reshape(n, n)

def sliding_window_inference(model, image_tensor, window_size=(512, 1024), stride_rate=0.5):
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

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
        root="./data/cityscapes",  
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )
    
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4) 

    model = Model(in_channels=3, n_classes=19, dino_fine_tune=False).to(device)
    
    checkpoint_path = "checkpoints/DINOv3 + unet-training V5/best_model-epoch=0014-val_loss=0.18933865303794542.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    num_classes = 19
    hist = torch.zeros((num_classes, num_classes), device=device)
    id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

    print("Running inference on validation set...")
    with torch.no_grad():
        from tqdm import tqdm
        for i, (images, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
            print(f"Processing batch {i+1}/{len(val_loader)}...")
            images = images.to(device)

            labels = labels.apply_(lambda x: id_to_trainid.get(x, 255)).long().squeeze(1)
            labels = labels.to(device)

            outputs = sliding_window_inference(
                model=model, 
                image_tensor=images, 
                window_size=(512, 1024), 
                stride_rate=0.5
            )

            predictions = outputs.argmax(dim=1)
            hist += fast_hist(labels.flatten(), predictions.flatten(), num_classes)

    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp

    results = {}
    total_iou = 0.0
    total_dice = 0.0

    for cat_name, class_ids in CATEGORY_MAPPING.items():
        cat_tp = tp[class_ids].sum().item()
        cat_fp = fp[class_ids].sum().item()
        cat_fn = fn[class_ids].sum().item()

        iou = cat_tp / (cat_tp + cat_fp + cat_fn) if (cat_tp + cat_fp + cat_fn) > 0 else 0.0
        dice = (2 * cat_tp) / (2 * cat_tp + cat_fp + cat_fn) if (2 * cat_tp + cat_fp + cat_fn) > 0 else 0.0

        results[f"Dice_{cat_name}"] = round(dice, 4)
        results[f"IoU_{cat_name}"] = round(iou, 4)

        total_iou += iou
        total_dice += dice

    results["MeanDice"] = round(total_dice / len(CATEGORY_MAPPING), 4)
    results["MeanIoU"] = round(total_iou / len(CATEGORY_MAPPING), 4)
    results["NumSamples"] = len(val_dataset)

    print("\n--- Final Metrics ---")
    print(json.dumps(results, indent=2))

    with open("final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()