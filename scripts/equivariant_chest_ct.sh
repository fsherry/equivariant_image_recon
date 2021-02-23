#!/bin/bash
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p gpu


train_sizes=(10 20 50 100 200 500 1000 2000 5000)
train_size=${train_sizes[$SLURM_ARRAY_TASK_ID]}
N_iterations=100000

python equivariant_chest_ct.py --N_train=$train_size --name=equivariant_ct_N=$train_size --N_epochs=$((N_iterations / train_size)) --depth=8 --save_path=$save_path --data_path=$data_path
