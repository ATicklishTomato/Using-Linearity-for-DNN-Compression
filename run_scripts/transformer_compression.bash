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

# Load modules or software if needed optimized for GPU use if available
# In the example PyTorch is made available for import in to my_sript.py
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -m venv .venv
source .venv/bin/activate # Activate virtual environment

pip install -r requirements.txt # Install leftover dependencies

# Execute the script or command
python main.py -m llama7b -d tinystories -e compression --verbose