#!/bin/bash

# Parameters
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=38
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=120G
#SBATCH --error=slurm_logs/log_%j.err
#SBATCH --job-name=print_env
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/log_%j.out
#SBATCH --signal=USR2@120
#SBATCH --time=30
#SBATCH --wckey=submitit

echo "Starting job..."

# command
srun --unbuffered \
    --output slurm_logs/log_%j_%t.out \
    --error slurm_logs/log_%j_%t.err \
    --exclusive --cpus-per-task=38 \
    /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
    /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py
