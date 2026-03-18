"""
This script provides and example implementation of a prediction pipeline
for a PyTorch U-Net model. It loads a pre-trained model, processes input
images, and saves the predicted segmentation masks.

You can use this file for submissions to the Challenge server. Customize
the `preprocess` and `postprocess` functions to fit your model's input
and output requirements.
"""

from pathlib import Path
from xml.parsers.expat import model

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
    transform = Compose(
        [
            ToImage(),
            # Resize((256, 512)),
            ToDtype(torch.float32, scale=True),
            Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # normalization values from ImageNet, since the DINO model is pretrained on ImageNet
        ]
    )

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    pred_max = torch.argmax(pred, dim=1, keepdim=True)

    prediction_numpy = pred_max.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Model()
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        state_dict,
        strict=True,  # Ensure the state dict matches the model architecture
    )
    model.eval().to(device)

    image_files = list(
        Path(IMAGE_DIR).glob("*.png")
    )  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = sliding_window_inference(
                model=model, 
                image_tensor=img_tensor, 
                window_size=(512, 1024), 
                stride_rate=0.5
            )

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
