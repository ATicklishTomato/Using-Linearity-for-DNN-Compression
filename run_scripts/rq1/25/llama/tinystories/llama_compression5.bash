#!/bin/bash

#SBATCH --job-name=llama_compression5
#SBATCH --output=llama_compression5_output_%j.txt
#SBATCH --partition tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Only load python because PyTorch module is garbage on this cluster
module purge
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate # Activate virtual environment

# Install dependencies
pip install -r requirements.txt

# Execute the script or command
python main.py -m llama-3-1b -d tinystories -e compression --max_batches 1024 --batch_size 4 --threshold "25%" --seed 1952 --save