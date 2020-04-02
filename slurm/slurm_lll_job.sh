#!/usr/bin/env bash

#SBATCH --time 3-0
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH -o logs/output-%j.out

cd /home/skoroki/zslll-master
python src/run.py $cli_args
