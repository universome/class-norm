#!/usr/bin/env bash

num_runs=$1

experiments_dir="aggregations"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub
case "$dataset" in
    cub) mem=64G ;;
    awa) mem=256G ;;
    *) mem=128G ;;
esac

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for aggregation_type in aggregate_logits aggregate_protos aggregate_losses; do
        for test_aggregation_type in aggregate_logits aggregate_protos; do
            for num_prototypes in 1 10 25; do
                cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                        "--config.hp.head.aggregation_type $aggregation_type" \
                        "--config.hp.head.test_aggregation_type $test_aggregation_type" \
                        "--config.hp.head.num_prototypes $num_prototypes")
                # echo "$cli_args"
                sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
            done
        done
    done
done
