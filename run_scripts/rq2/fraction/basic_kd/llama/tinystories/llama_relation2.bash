#!/bin/bash

#SBATCH --job-name=llama_relation_bkd2
#SBATCH --output=llama_relation_bkd2_output_%j.txt
#SBATCH --partition tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Only load python because PyTorch module is garbage on this cluster
module purge
module load Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate # Activate virtual environment

# Install dependencies
pip install -r requirements.txt

# Execute the script or command
python main.py -m llama-3-1b -d tinystories -e relation -l fraction --relation basic_kd --batch_size 4 --seed 762 --save