# NNCV Final Assignment - Cityscapes Semantic Segmentation

**Course:** 5LSM0 Neural Networks for Computer Vision  
**Author:** Martijn van de Sande (1730193)  
**Institution:** Eindhoven University of Technology  

This repository contains the code used for my final assignment in 5LSM0. Two segmentation models are trained on a Cityscapes-based dataset provided for the course.

1. **D3-Unet-v5** (`DINOv3-Unet_V5` branch) - A U-Net extended with multi-scale DINOv3-ViT-B/16 feature fusion, ASPP bottleneck, Squeeze-and-Excitation blocks, and sliding-window TTA inference. Achieves Mean Dice **0.566** on the course server.
2. **enet v2** (`eff-Unet-V2` branch) - A compact distillation network where a frozen SegFormer-B5 teacher is distilled into an EfficientViT-B0 student using logit-level KD. Achieves **94.7 FPS** at **2.98 MB** and Mean Dice **0.417**.

> **Note on pretrained weights:** Pretrained weights are not included in this repository because the checkpoint files are too large for standard git tracking. Download links and expected file locations are listed in the setup section below.

> **Note on branches:** Make sure you are on the right branch before training or evaluating. The `main` branch only has the course baseline. Each model version has its own branch. `DINOv3-Unet_V6` and `eff-Unet-V3` are extra branches used for testing and are not part of the report.

---

## Repository Structure

```text
NNCV_Martijn/
├── Final assignment/          # All assignment code lives here
│   ├── train.py               # Training script
│   ├── model.py               # Model architecture
│   ├── evaluate.py            # Evaluation script
│   ├── predict.py             # Inference script
│   ├── predict_ood.py         # Out-of-distribution prediction
│   ├── jobscript_slurm.sh     # Snellius SLURM job script
│   ├── main.sh                # Called by jobscript, runs train.py with settings
│   ├── main_eval.sh           # Runs evaluation
│   ├── eval.sh                # Evaluation shell script
│   ├── download_docker_and_data.sh  # Data download helper
│   ├── Dockerfile             # Docker environment
├── Weekly notebooks/          # Course weekly exercises (not part of the report)
```

---

## Branch Overview

Each branch is one development iteration. Always check out the correct branch before running anything.

| Branch         | Model   | Description                                             | In report? |
| -------------- | ------- | ------------------------------------------------------- | ---------- |
| main           | D3-Unet | V1 - DINOv3 added to the baseline, no scheduler yet     | Yes        |
| DINOv3-Unet_V2 | D3-Unet | CosineAnnealingLR and Dice loss added                   | Yes        |
| DINOv3-Unet_V3 | D3-Unet | Added SEB, ASPP, augmentation (had a normalisation bug) | Yes        |
| DINOv3-Unet_V4 | D3-Unet | Normalisation fixed, TTA flipping added                 | Yes        |
| DINOv3-Unet_V5 | D3-Unet | Full-resolution sliding-window inference, main result   | Yes        |
| DINOv3-Unet_V6 | D3-Unet | Extra testing, not used in the report                   | No         |
| segB5-Unet-V6  | D3-Unet | Extra testing, not used in the report                   | No         |
| eff-Unet-V1    | enet    | MobileNetV3 backbone, turned out too large              | Yes        |
| eff-Unet-V2    | enet    | EfficientViT-B0 backbone, main enet result              | Yes        |
| eff-Unet-V3    | enet    | Extra testing, not used in the report                   | No         |

---

## Running on Snellius

All training was done on the [Snellius](https://www.surf.nl/en/dutch-national-supercomputer-snellius) national supercomputer. The setup uses Apptainer to run a container, so there is no conda environment to set up.

### 1. Clone and check out the right branch

```bash
git clone https://github.com/Martijn-vd-s/NNCV_Martijn.git
cd NNCV_Martijn

# For D3-Unet-v5:
git checkout DINOv3-Unet_V5

# For enet:
git checkout eff-Unet-V2
```

### 2. Download pretrained weights

**DINOv3-ViT-B/16** (used in D3-Unet-v5)  
Get it from the official repo: https://github.com/facebookresearch/dinov3  
Follow the instructions there to download the `dinov3_vitb16` checkpoint and place it in `Final assignment/`.

**EfficientViT-B0** (used in enet)  
Get it from: https://github.com/CVHub520/efficientvit  
Place the `b0.pt` checkpoint in `Final assignment/`.

**SegFormer-B5 teacher** (used in enet)  

```bash
python -c "
from transformers import SegformerForSemanticSegmentation
SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
    cache_dir='Final assignment/mit-b5'
)"
```

### 3. Get the dataset

```bash
cd "Final assignment"
bash download_docker_and_data.sh
```

### 4. Submit the training job

```bash
cd "Final assignment"
sbatch jobscript_slurm.sh
```

This submits the training job on Snellius with an A100 GPU and the configured time limit. The job script then calls `main.sh` which runs `train.py` with all the training settings. No further configuration is needed.

---

## Evaluation

```bash
cd "Final assignment"
bash main_eval.sh
```

This reports Mean Dice averaged over the seven category groups from the course server: flat, construction, object, nature, sky, human, and vehicle.

---

## Results

| Model | mDice | mIoU | Size (MB) | GFLOPs | FPS |
|-------|-------|------|-----------|--------|-----|
| Baseline U-Net | 0.478 | 0.392 | 65.93 | 2563.6 | 8.83 |
| D3-Unet-v5 | 0.566 | 0.460 | 431.3 | 4127.1 | 2.01 |
| enet v2 | 0.417 | 0.324 | 2.98 | 10.79 | 94.72 |

All results are from the course submission server on 500 test images.

---
## Leaderboard Submissions

| Benchmark | Team name | Model |
|---|---|---|
| Peak performance | MvdS_D3-Unet-v5_Peak | D3-Unet-v5 (branch: DINOv3-Unet_V5) |
| Efficiency | MvdS_eNet_V1.6 | enet v2 (branch: eff-Unet-V2) |

**Author:** Martijn van de Sande  
**Student number:** 1730193  
**TU/e email:** m.v.d.sande@student.tue.nl

---
## Notes

- D3-Unet-v5 completed 16 out of 100 scheduled epochs due to compute time limit in the slurm script.  Further training would likely give small gains and was not considered worth the additional compute given the already stable behaviour. The best checkpoint by validation loss was used for submission. 
- `DINOv3-Unet_V6`, `segB5-Unet-V6`, and `eff-Unet-V3` are extra experiments for testing and not part of the report.
