#!/usr/bin/env bash

num_runs=$1

experiments_dir="multi_proto_random_optim"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub
case "$dataset" in
    cub) mem=64G ;;
    awa) mem=256G ;;
    *) mem=128G ;;
esac

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for lr in 0.001 0.003 0.01; do
        for optim_type in sgd adam; do
            for batch_size in 10 25; do
                for max_num_epochs in 5 10 25; do
                    for num_prototypes in 10 25; do
                        for num_test_prototypes in 25 100; do
                            cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                                    "--config.hp.head.num_prototypes $num_prototypes" \
                                    "--config.hp.head.num_test_prototypes $num_test_prototypes" \
                                    "--config.hp.head.noise.transform_layers {128-256-128}" \
                                    "--config.hp.head.noise.same_for_each_class false" \
                                    "--config.hp.head.fusing_type full_mult_int" \
                                    "--config.hp.head.attrs_transform_layers {312}" \
                                    "--config.hp.optim.type $optim_type" \
                                    "--config.hp.optim.kwargs.lr $lr" \
                                    "--config.hp.batch_size $batch_size" \
                                    "--config.hp.max_num_epochs $max_num_epochs" \
                                    "--config.hp.head.after_fuse_transform_layers {256-512}" \
                                    "--config.hp.head.noise.std 0.1")
                            # echo "$cli_args"
                            sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                        done
                    done

                    # Baseline
                    cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                            "--config.hp.head.num_prototypes 1" \
                            "--config.hp.head.num_test_prototypes 1" \
                            "--config.hp.head.noise.transform_layers {128}" \
                            "--config.hp.head.noise.same_for_each_class true" \
                            "--config.hp.head.fusing_type concat" \
                            "--config.hp.head.attrs_transform_layers {312}" \
                            "--config.hp.head.after_fuse_transform_layers {512}" \
                            "--config.hp.optim.type $optim_type" \
                            "--config.hp.optim.kwargs.lr $lr" \
                            "--config.hp.batch_size $batch_size" \
                            "--config.hp.max_num_epochs $max_num_epochs" \
                            "--config.hp.head.noise.std 0.0")
                    # echo "$cli_args"
                    sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                done
            done
        done
    done
done
