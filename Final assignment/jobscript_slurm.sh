#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --job-name=dinov3-Unet   
#SBATCH --output=logs/%x_%j_%(%Y-%m-%d_%H-%M)T.out
#SBATCH --error=logs/%x_%j_%(%Y-%m-%d_%H-%M)T.err    

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh