#!/bin/bash

#SBATCH --job-name=resnet_compression1
#SBATCH --output=resnet_compression1_output_%j.txt
#SBATCH --partition tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G

# Only load python because PyTorch module is garbage on this cluster
module purge
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate # Activate virtual environment

# Install dependencies
pip install -r requirements.txt

# Execute the script or command
python main.py -m resnet50 -d cifar10 -e linear_approximator_compression -l fraction --threshold "75%" --epochs 20 --seed 42 --save --skip_finetune