#!/bin/bash

#SBATCH --job-name=llama_relation_bkd2
#SBATCH --output=llama_relation_bkd2_output_%j.txt
#SBATCH --partition gpu_a100
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G

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
python main.py -m llama-2-7b -d superglue --batch_size 4 -e relation -l procrustes --relation basic_kd --seed 762 --save --skip_finetune