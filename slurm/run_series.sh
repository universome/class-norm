#!/usr/bin/env bash

num_runs=$1

experiments_dir="random_proto_embs"
config="multi_proto"
for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for num_prototypes in 1 10; do
        for dataset in cub; do
            for transform_layers in "64" "64,128" "64,128,256"; do
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

                    cli_args="--config.hp.head.num_prototypes $num_prototypes\
                            --config.hp.head.context.transform_layers $transform_layers\
                            --config.hp.head.scale_value $scale_value\
                            -c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed"
                    # sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                    echo $cli_args
                done
            done
        done
    done
done
