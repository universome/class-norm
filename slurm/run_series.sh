#!/usr/bin/env bash

num_runs=$1

# Multi-prototypical training
experiments_dir="multi_proto_experiments"
config="multi_proto"
for push_protos_apart in true false; do
    for num_prototypes in 1 2 5 10; do
        for scale_value in 1 3 5; do
            for dataset in cub; do
                cli_args="--config.hp.head.num_prototypes $num_prototypes\
                        --config.hp.head.scale_value $scale_value\
                        --config.hp.push_protos_apart.enabled $push_protos_apart\
                        -c $config -d $dataset --experiments_dir $experiments_dir"
                sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                # echo $cli_args
            done
        done
    done
done
