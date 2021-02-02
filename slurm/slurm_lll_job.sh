#!/usr/bin/env bash

mkdir -p /tmp/skoroki/datasets
cp "/ibex/scratch/skoroki/datasets/${dataset_full_name}.tar.gz" /tmp/skoroki/datasets
tar zxvf "/tmp/skoroki/datasets/${dataset_full_name}.tar.gz" -C /tmp/skoroki/datasets

echo "`gpustat`"
echo "`nvidia-smi`"
echo "CLI args: $cli_args"

cd $project_dir
python src/run.py $cli_args
