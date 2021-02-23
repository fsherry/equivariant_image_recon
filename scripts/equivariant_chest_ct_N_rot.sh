#!/bin/bash
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p gpu


train_size=100
N_rots=(2 3 4 6 8 12)
N_rot=${N_rots[$SLURM_ARRAY_TASK_ID]}
N_iterations=100000
channels=$((96/$N_rot))

python equivariant_chest_ct.py --N_train=$train_size --name=equivariant_ct_N_rot=$N_rot --N_rot=$N_rot --channels=$channels --N_epochs=$((N_iterations / train_size)) --depth=8 --save_path=$save_path --data_path=$data_path
