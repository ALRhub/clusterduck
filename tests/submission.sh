#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --error=/home/i53/mitarbeiter/balazs/repos/clusterduck/tests/logs/%j_log.err
#SBATCH --gres=gpu:4
#SBATCH --job-name=print_env
#SBATCH --mem-per-cpu=15000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append
#SBATCH --output=/home/i53/mitarbeiter/balazs/repos/clusterduck/tests/logs/%j_log.out
#SBATCH --signal=USR2@120
#SBATCH --time=5
#SBATCH --wckey=submitit

# command
srun --unbuffered \
    --output /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/logs/%j_log.out \
    --error /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/logs/%j_log.err \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python \
    -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py
