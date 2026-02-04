#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition tue.gpu2.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# Load modules or software if needed optimized for GPU use if available
# In the example PyTorch is made available for import in to my_sript.py
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1


source .venv/bin/activate # Activate virtual environment

pip install -r requirements.txt # Install leftover dependencies

# Execute the script or command
python my_script.py