#!/usr/bin/env bash

num_runs=$1

experiments_dir="softmax_mean"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub
case "$dataset" in
    cub) mem=64G ;;
    awa) mem=256G ;;
    *) mem=128G ;;
esac

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for logits_aggregation_type in mean softmax_mean; do
        for scale_value in 10 11 12; do
            for num_prototypes in 1 10 25; do
                cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                        "--config.hp.head.logits_aggregation_type $logits_aggregation_type" \
                        "--config.hp.head.scale.value $scale_value" \
                        "--config.hp.head.num_prototypes $num_prototypes")
                # echo "$cli_args"
                sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
            done
        done
    done
done
