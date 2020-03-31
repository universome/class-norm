#!/usr/bin/env bash

num_runs=$1

experiments_dir="random_proto_embs"
# logs_dir="${experiments_dir}-logs"
mkdir -p $experiments_dir
# mkdir -p $logs_dir
config="multi_proto"
for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for num_prototypes in 1; do
        for dataset in cub; do
            for transform_layers in {64} {64-128} {64-128-256}; do
                for std in 0 0.1 0.5 1; do
                    case "$dataset" in
                        cub) mem=64G ;;
                        awa) mem=256G ;;
                        *) mem=128G ;;
                    esac

                    case "$num_prototypes" in
                        10) scale_value=5 ;;
                        *) scale_value=10 ;;
                    esac

                    cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                            "--config.hp.head.num_prototypes $num_prototypes" \
                            "--config.hp.head.context.transform_layers $transform_layers" \
                            "--config.hp.head.context.std $std" \
                            "--config.hp.head.scale_value $scale_value")
                    # echo "$cli_args"
                    sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                done
            done
        done
    done
done
