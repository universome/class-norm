#!/usr/bin/env bash

num_runs=$1

experiments_dir="multi_proto_random"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for num_prototypes in 1 10; do
        for num_test_prototypes in 1 10; do
            for transform_layers in {64} {64-128}; do
                for scale_value in 10; do
                    for std in 0 0.1; do
                        case "$dataset" in
                            cub) mem=64G ;;
                            awa) mem=256G ;;
                            *) mem=128G ;;
                        esac

                        # case "$num_prototypes" in
                        #     10) scale_value=5 ;;
                        #     *) scale_value=10 ;;
                        # esac

                        cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                                "--config.hp.head.num_prototypes $num_prototypes" \
                                "--config.hp.head.num_test_prototypes $num_test_prototypes" \
                                "--config.hp.head.context.transform_layers $transform_layers" \
                                "--config.hp.head.scale_value $scale_value" \
                                "--config.hp.head.context.std $std")
                        echo "$cli_args"
                        # sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                    done
                done
            done
        done
    done
done

# experiments_dir="multi_headed"
# mkdir -p $experiments_dir
# config="multi_proto"
# dataset=cub

# for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#     for num_prototypes in 1 10 25; do
#         for embedder_hidden_layers in {128} {128-256} {128-256-256}; do
#             for use_final_activation in true false; do
#                 for scale_value in 5 10 15; do
#                     case "$dataset" in
#                         cub) mem=64G ;;
#                         awa) mem=256G ;;
#                         *) mem=128G ;;
#                     esac

#                     # case "$num_prototypes" in
#                     #     10) scale_value=5 ;;
#                     #     *) scale_value=10 ;;
#                     # esac

#                     cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
#                             "--config.hp.head.num_prototypes $num_prototypes" \
#                             "--config.hp.head.embedder_hidden_layers $embedder_hidden_layers" \
#                             "--config.hp.head.scale_value $scale_value" \
#                             "--config.hp.head.use_final_activation $use_final_activation")
#                     # echo "$cli_args"
#                     sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#                 done
#             done
#         done
#     done
# done
