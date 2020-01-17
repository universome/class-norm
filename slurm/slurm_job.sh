#!/usr/bin/env bash

#SBATCH --time 10080
#SBATCH --mem 64G
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task 4
#SBATCH -o logs/output-%j.out

cd /home/skoroki/zslll
# python src/run.py $cli_args
firelab start configs/classifier.yml
