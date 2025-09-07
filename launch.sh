#!/bin/bash
#SBATCH --job-name=array
#SBATCH --output=./saves/experiment-%a/train.out
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00

source /home/erfan/pyenv/bin/activate

export SAVE_PATH="./saves/experiment-$SLURM_ARRAY_TASK_ID"
mkdir -p "$SAVE_PATH"

HYPER_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" hyperparams.txt)
export $HYPER_LINE

srun --unbuffered python3 train_fixed_taskset.py
