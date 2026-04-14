#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=05:00:00
#SBATCH --job-name=eff-Unet-V2.1
#SBATCH --output=logs/%x_%j_.out
#SBATCH --error=logs/%x_%j_.err    

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh