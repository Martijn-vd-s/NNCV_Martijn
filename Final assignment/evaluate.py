import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype, InterpolationMode
from model import Model

# Mapping the 19 Cityscapes train_ids into the 7 Server Categories
CATEGORY_MAPPING = {
    "Flat": [0, 1],               # road, sidewalk
    "Construction": [2, 3, 4],    # building, wall, fence (Matched your JSON spelling!)
    "Object": [5, 6, 7],          # pole, traffic light, traffic sign
    "Nature": [8, 9],             # vegetation, terrain
    "Sky": [10],                  # sky
    "Human": [11, 12],            # person, rider
    "Vehicle": [13, 14, 15, 16, 17, 18] # car, truck, bus, train, motorcycle, bicycle
}

def fast_hist(a, b, n):
    """Calculates the confusion matrix for a single batch."""
    k = (a >= 0) & (a < n)
    return torch.bincount(n * a[k] + b[k], minlength=n ** 2).reshape(n, n)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # --- 1. Load the Validation Dataset ---
    img_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    target_transform = Compose([
        ToImage(),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    val_dataset = Cityscapes(
        root="./data/cityscapes", # to right path!!!
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # --- 2. Load the Model & Best Weights ---
    model = Model(in_channels=3, n_classes=19, dino_fine_tune=False).to(device)
    
    # to checkpoint path !!!!
    checkpoint_path = "checkpoints/DINOv3 + unet-training V4/best_model-epoch=0076-val_loss=0.2837434709072113.pt" 
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # --- 3. Run Inference and Accumulate Pixels ---
    num_classes = 19
    hist = torch.zeros((num_classes, num_classes), device=device)
    
    id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            # 1. Map the IDs to Train IDs while it is STILL ON THE CPU
            labels = labels.apply_(lambda x: id_to_trainid.get(x, 255)).long().squeeze(1)
            
            # 2. NOW move it to the GPU
            labels = labels.to(device)

            # get predictions from normal images
            outputs_normal = model(images)
            # get predictions from flipped images
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            outputs_flipped = torch.flip(outputs_flipped, dims=[3])

            # Average the outputs from normal and flipped images
            outputs = (outputs_normal + outputs_flipped) / 2.0

            # Get predicted class for each pixel
            predictions = outputs.argmax(dim=1)
            
            # Accumulate confusion matrix
            hist += fast_hist(labels.flatten(), predictions.flatten(), num_classes)

    # --- 4. Calculate True Positives, False Positives, False Negatives ---
    tp = torch.diag(hist)
    fp = hist.sum(dim=0) - tp
    fn = hist.sum(dim=1) - tp

    # --- 5. Group into Categories and Calculate Metrics ---
    results = {}
    total_iou = 0.0
    total_dice = 0.0

    for cat_name, class_ids in CATEGORY_MAPPING.items():
        # Sum TP, FP, FN for all classes in this category
        cat_tp = tp[class_ids].sum().item()
        cat_fp = fp[class_ids].sum().item()
        cat_fn = fn[class_ids].sum().item()

        # Avoid division by zero
        iou = cat_tp / (cat_tp + cat_fp + cat_fn) if (cat_tp + cat_fp + cat_fn) > 0 else 0.0
        dice = (2 * cat_tp) / (2 * cat_tp + cat_fp + cat_fn) if (2 * cat_tp + cat_fp + cat_fn) > 0 else 0.0

        results[f"Dice{cat_name}"] = dice
        results[f"IoU{cat_name}"] = iou
        
        total_iou += iou
        total_dice += dice

    # Calculate overall means
    results["MeanDice"] = total_dice / len(CATEGORY_MAPPING)
    results["MeanIoU"] = total_iou / len(CATEGORY_MAPPING)
    results["NumSamples"] = len(val_dataset)

    # --- 6. Print and Save JSON ---
    print("\n--- Final Metrics ---")
    print(json.dumps(results, indent=2))
    
    with open("final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()