#!/usr/bin/env bash

num_runs=$1

experiments_dir="multi_proto_random"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub
case "$dataset" in
    cub) mem=64G ;;
    awa) mem=256G ;;
    *) mem=128G ;;
esac

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for noise_transform_layers in {128} {128-256-128}; do
        for attrs_transform_layers in {312} {312-256}; do
            for after_fuse_transform_layers in {512} {256-512}; do
                for same_for_each_class in true false; do
                    for fusing_type in concat full_mult_int; do
                        for std in 0.1 0.5; do
                            cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                                    "--config.hp.head.num_prototypes 10" \
                                    "--config.hp.head.num_test_prototypes 25" \
                                    "--config.hp.head.noise.transform_layers $noise_transform_layers" \
                                    "--config.hp.head.noise.same_for_each_class $same_for_each_class" \
                                    "--config.hp.head.fusing_type $fusing_type" \
                                    "--config.hp.head.attrs_transform_layers $attrs_transform_layers" \
                                    "--config.hp.head.after_fuse_transform_layers $after_fuse_transform_layers" \
                                    "--config.hp.head.noise.std $std")
                            # echo "$cli_args"
                            sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                        done
                    done
                done
            done
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
            "--config.hp.head.noise.std 0.0")
    # echo "$cli_args"
    sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
done
