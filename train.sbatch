#!/bin/bash
#SBATCH --job-name="mover_lstm"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
eval "$(conda shell.bash hook)"
conda activate thesis
cd mover/
srun python train_gan.py
