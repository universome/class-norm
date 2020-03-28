#!/usr/bin/env bash

num_runs=$1

# Multi-prototypical training
experiments_dir="logits_scaling"
config="multi_proto"
for num_prototypes in 10; do
    for logits_scale_value in 0.5 1 2 5 10; do
        for dataset in cub; do
            cli_args="--config.hp.head.num_prototypes $num_prototypes\
                      --config.hp.logits_scaling.scale_value $logits_scale_value\
                      -c $config -d $dataset --experiments_dir $experiments_dir"
            sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
            # echo $cli_args
        done
    done
done
