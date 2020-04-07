#!/usr/bin/env bash

num_runs=$1

experiments_dir="mph_random_optim"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub
case "$dataset" in
    cub) mem=64G ;;
    awa) mem=256G ;;
    *) mem=128G ;;
esac

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for lr in 0.0001 0.0005 0.001 0.0025 0.005 0.01; do
        for momentum in 0.0 0.1 0.25 0.4 0.5 0.6 0.75 0.9 0.95 0.99; do
                cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                        "--config.hp.optim.kwargs.lr $lr" \
                        "--config.hp.optim.kwargs.momentum $momentum")
                # echo "$cli_args"
                sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
        done
    done
done
