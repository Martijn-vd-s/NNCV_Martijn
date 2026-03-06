#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=3:00:00

# Pull container from dockerhub
apptainer pull container.sif docker://cclaess/5lsm0:v1

# Use the huggingface-cli package inside the container to download the data
mkdir -p data
apptainer exec container.sif \
    huggingface-cli download TimJaspersTue/5LSM0 --local-dir ./data --repo-type dataset --max-worker 1 
