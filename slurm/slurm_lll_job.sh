#!/usr/bin/env bash

#SBATCH --time 3-0
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4

mkdir -p /tmp/skoroki/czsl/data
cp "/ibex/scratch/skoroki/datasets/${dataset_full_name}.tar.gz" /tmp/skoroki/czsl/data
tar zxvf "/tmp/skoroki/czsl/data/${dataset_full_name}.tar.gz" -C /tmp/skoroki/czsl/data

cd /home/skoroki/zslll-master
python src/run.py $cli_args
