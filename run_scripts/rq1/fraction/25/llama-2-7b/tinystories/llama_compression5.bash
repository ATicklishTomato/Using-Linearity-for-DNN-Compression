#!/bin/bash

#SBATCH --job-name=llama_compression5
#SBATCH --output=llama_compression5_output_%j.txt
#SBATCH --partition gpu_a100
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G

# Only load python because PyTorch module is garbage on this cluster
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

mkdir -p /scratch-shared/lwubben/data
python -m venv .venv
source .venv/bin/activate # Activate virtual environment

# Install dependencies
pip install -r requirements.txt

# Execute the script or command
python main.py -m llama-2-7b -d tinystories --batch_size 4 -e compression -l fraction --threshold "25%" --seed 1952 --save --skip_finetune