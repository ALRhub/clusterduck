#!/bin/bash

# Parameters
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --error=slurm_logs/log_%j.err
#SBATCH --exclude='node[1-5]'
#SBATCH --job-name=print_env
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/log_%j.out
#SBATCH --signal=USR2@120
#SBATCH --time=2880
#SBATCH --wckey=submitit

echo "Starting job..."

# command
# srun --unbuffered \
#     --output slurm_logs/log_%j_%t.out \
#     --error slurm_logs/log_%j_%t.err \
#     --slurmd-debug=5 \
#     --exclusive --exact \
#     --gpus=4 --gpus-per-task=1 \
#     /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
#     /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py


srun --unbuffered \
    --output slurm_logs/log_%j_0.out \
    --error slurm_logs/log_%j_0.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

srun --unbuffered \
    --output slurm_logs/log_%j_1.out \
    --error slurm_logs/log_%j_1.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

srun --unbuffered \
    --output slurm_logs/log_%j_2.out \
    --error slurm_logs/log_%j_2.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

srun --unbuffered \
    --output slurm_logs/log_%j_3.out \
    --error slurm_logs/log_%j_3.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

srun --unbuffered \
    --output slurm_logs/log_%j_5.out \
    --error slurm_logs/log_%j_5.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

srun --unbuffered \
    --output slurm_logs/log_%j_6.out \
    --error slurm_logs/log_%j_6.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

srun --unbuffered \
    --output slurm_logs/log_%j_7.out \
    --error slurm_logs/log_%j_7.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

srun --unbuffered \
    --output slurm_logs/log_%j_8.out \
    --error slurm_logs/log_%j_8.err \
    --slurmd-debug=5 \
    -n1 --gres=gpu:1 --exclusive \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/.env/bin/python -u \
    /home/i53/mitarbeiter/balazs/repos/clusterduck/tests/print_env.py &

wait
