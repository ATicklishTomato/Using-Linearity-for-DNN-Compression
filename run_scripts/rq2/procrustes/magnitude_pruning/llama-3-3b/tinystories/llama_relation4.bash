#!/bin/bash

#SBATCH --job-name=llama_relation_mag4
#SBATCH --output=llama_relation_mag4_output_%j.txt
#SBATCH --partition gpu_mig
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Only load python because PyTorch module is garbage on this cluster
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -m venv .venv
source .venv/bin/activate # Activate virtual environment

# Install dependencies
pip install -r requirements.txt

# Execute the script or command
python main.py -m llama-3-3b -d tinystories -e relation -l procrustes --relation magnitude_pruning --seed 1843 --save