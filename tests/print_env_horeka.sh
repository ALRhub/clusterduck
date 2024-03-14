#!/bin/bash

# Parameters
#SBATCH --partition=accelerated
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
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
    --exclusive --exact \
    /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
    /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py


# srun --unbuffered \
#     --output slurm_logs/log_%j_0.out \
#     --error slurm_logs/log_%j_0.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# srun --unbuffered \
#     --output slurm_logs/log_%j_1.out \
#     --error slurm_logs/log_%j_1.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# srun --unbuffered \
#     --output slurm_logs/log_%j_2.out \
#     --error slurm_logs/log_%j_2.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# srun --unbuffered \
#     --output slurm_logs/log_%j_3.out \
#     --error slurm_logs/log_%j_3.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# srun --unbuffered \
#     --output slurm_logs/log_%j_5.out \
#     --error slurm_logs/log_%j_5.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# srun --unbuffered \
#     --output slurm_logs/log_%j_6.out \
#     --error slurm_logs/log_%j_6.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# srun --unbuffered \
#     --output slurm_logs/log_%j_7.out \
#     --error slurm_logs/log_%j_7.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# srun --unbuffered \
#     --output slurm_logs/log_%j_8.out \
#     --error slurm_logs/log_%j_8.err \
#     -n1 --gres=gpu:1 --exclusive \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/.env/bin/python -u \
#     /home/hk-project-robolrn/jh8109/repos/clusterduck/tests/print_env.py &

# wait
