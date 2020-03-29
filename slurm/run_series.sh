#!/usr/bin/env bash

num_runs=$1

# Multi-prototypical training
experiments_dir="senet"
config="multi_proto"
for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for num_prototypes in 1 10; do
        for senet_enabled in true false; do
            for dataset in cub; do
                case "$dataset" in
                    cub) mem=64G ;;
                    awa) mem=256G ;;
                    *) mem=128G ;;
                esac

                case "$num_prototypes" in
                    10) scale_value=5 ;;
                    *) scale_value=1 ;;
                esac

                cli_args="--config.hp.head.num_prototypes $num_prototypes\
                        --config.hp.head.senet.enabled $senet_enabled\
                        --config.hp.head.scale_value $scale_value\
                        -c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed"
                sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                # echo $cli_args
                # echo $mem
            done
        done
    done
done

# experiments_dir="gaussian_noise"
# config="multi_proto"
# for num_prototypes in 3 10; do
#     for z_size in 64; do
#         for std in 0.01 0.1 1.; do
#             for fusing_type in concat pure_mult_int full_mult_int; do
#                 for dataset in cub; do
#                     cli_args="--config.hp.head.num_prototypes $num_prototypes\
#                             --config.hp.head.context.type gaussian_noise\
#                             --config.hp.head.context.z_size $z_size\
#                             --config.hp.head.context.proto_emb_size $z_size\
#                             --config.hp.head.context.std $std\
#                             --config.hp.head.fusing_type $fusing_type\
#                             -c $config -d $dataset --experiments_dir $experiments_dir"
#                     sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#                     # echo $cli_args
#                 done
#             done
#         done
#     done
# done

# experiments_dir="transformed_gaussian_noise"
# config="multi_proto"
# for num_prototypes in 1 3 10; do
#     for z_size in 32 256; do
#         for std in 0.01 0.1 0.5 1. 2; do
#             for dataset in cub; do
#                 cli_args="--config.hp.head.num_prototypes $num_prototypes\
#                           --config.hp.head.context.type gaussian_noise\
#                           --config.hp.head.context.z_size $z_size\
#                           --config.hp.head.context.std $std\
#                         -c $config -d $dataset --experiments_dir $experiments_dir"
#                 sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#                 # echo $cli_args
#             done
#         done
#     done
# done