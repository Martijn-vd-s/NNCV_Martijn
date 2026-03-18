#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:20:00
#SBATCH --job-name=Eval-DINOv3 
#SBATCH --output=logs/%x_%j_.out
#SBATCH --error=logs/%x_%j_.err    

# Execute the evaluation script inside the Apptainer container
srun apptainer exec --nv --env-file .env container.sif /bin/bash main_eval.sh