#!/usr/bin/env bash

#SBATCH --time 10080
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH -o slurm_output/output-%a.out

cd /users/skoroki/zslll
python src/run.py $cli_args
