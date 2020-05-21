#!/usr/bin/env bash

#SBATCH --time 3-0
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4

mkdir -p /tmp/skoroki/czsl/data
cp /home/skoroki/zslll/data/CUB_200_2011.tar.gz /tmp/skoroki/czsl/data/CUB_200_2011.tar.gz
tar zxvf /tmp/skoroki/czsl/data/CUB_200_2011.tar.gz -C /tmp/skoroki/czsl/data

cd /home/skoroki/zslll-master
python src/run.py $cli_args
