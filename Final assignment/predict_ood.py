"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import Model
import os
import csv

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"


def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # For example, resizing, normalization, etc.
    # Return a tensor suitable for model input
    transform = Compose([
        ToImage(),
        Resize(size=(256, 256), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.5,), std=(0.5,)),
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # Get the class with the highest probability
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("**/*.png"))

    print(f"Found {len(image_files)} images to process.")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open CSV file for predictions
    csv_path = Path(OUTPUT_DIR) / "predictions.csv"
    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass - model should return segmentation and inclusion decision
            seg_pred, include_decision = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(seg_pred, original_shape)

            # Create mirrored output folder structure
            relative_path = img_path.relative_to(IMAGE_DIR)
            out_path = Path(OUTPUT_DIR) / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            seg_pred_img = Image.fromarray(seg_pred.astype(np.uint8))
            seg_pred_img.save(out_path)

            # Record prediction
            predictions.append({
                'image_name': str(relative_path).replace('\\', '/'),
                'include': bool(include_decision)
            })

    # Write predictions to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'include'])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")

if __name__ == "__main__":
    main()