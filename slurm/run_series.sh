#!/usr/bin/env bash

num_runs=$1

experiments_dir="dropout_experiments"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub
case "$dataset" in
    cub) mem=64G ;;
    awa) mem=256G ;;
    *) mem=128G ;;
esac

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for aggregation_type in aggregate_logits aggregate_protos; do
        for dropout_type in attribute_wise element_wise; do
            for p in 0.0 0.05 0.1 0.25 0.5 0.75; do
                for attrs_transform_layers in 312-512 312-256-512; do
                    for num_prototypes in 1 10 25; do
                        cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                                "--config.hp.head.aggregation_type $aggregation_type" \
                                "--config.hp.head.dropout.type $dropout_type" \
                                "--config.hp.head.dropout.p $p" \
                                "--config.hp.head.attrs_transform_layers $attrs_transform_layers" \
                                "--config.hp.head.num_prototypes $num_prototypes")
                        # echo "$cli_args"
                        sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                    done
                done
            done
        done
    done
done
