#!/bin/bash

#SBATCH --job-name=transformer_compression
#SBATCH --output=transformer_compression_output_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=1                      # This is how to request a GPU

# Only load python because PyTorch module is garbage on this cluster
module purge
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate # Activate virtual environment

# Install dependencies
pip install -r requirements.txt

# Execute the script or command
python main.py -m llama7b -d tinystories -e compression --verbose