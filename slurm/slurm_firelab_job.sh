#!/usr/bin/env bash

#SBATCH --time 1-0
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 2

mkdir -p /tmp/skoroki/czsl/data
cp -r "/ibex/scratch/skoroki/datasets/${dataset}_feats" /tmp/skoroki/czsl/data

echo "`gpustat`"
echo "`nvidia-smi`"
echo "CLI args: $cli_args"

cd /home/skoroki/zslll-master
firelab start configs/zsl.yml $cli_args
