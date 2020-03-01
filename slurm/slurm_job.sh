#!/usr/bin/env bash

#SBATCH --time 7-0
#SBATCH --mem 64G
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task 4
#SBATCH -o logs/output-%j.out

cd /home/skoroki/zslll-master
python src/run.py $cli_args
# python src/run.py
# firelab start configs/classifier.yml $cli_args
