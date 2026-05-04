#!/bin/bash

#SBATCH --job-name=resnet_compression2
#SBATCH --output=resnet_compression2_output_%j.txt
#SBATCH --partition tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G

# Only load python because PyTorch module is garbage on this cluster
module purge
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate # Activate virtual environment

# Install dependencies
pip install -r requirements.txt

# Execute the script or command
python main.py -m resnet18 -d imagenet -e compression --max_batches 1024 --threshold "75%" --seed 762 --save --skip_finetune