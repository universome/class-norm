#!/usr/bin/env bash

num_runs=$1

experiments_dir="senet"
config="multi_proto"
for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for num_prototypes in 1 10; do
        for dataset in cub; do
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
                    --config.hp.head.senet.enabled false\
                    --config.hp.head.scale_value $scale_value\
                    -c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed"
            sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
            # echo $cli_args
        done
    done
done
for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for num_prototypes in 1 10; do
        for reduction_dim in 16 64 256; do
            for dataset in cub; do
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
                        --config.hp.head.senet.enabled true\
                        --config.hp.head.senet.reduction_dim $reduction_dim\
                        --config.hp.head.scale_value $scale_value\
                        -c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed"
                sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                # echo $cli_args
            done
        done
    done
done
