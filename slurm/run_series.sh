#!/usr/bin/env bash

num_runs=$1

experiments_dir="multi_proto_random_incremental"
mkdir -p $experiments_dir
config="multi_proto"
dataset=cub
case "$dataset" in
    cub) mem=64G ;;
    awa) mem=256G ;;
    *) mem=128G ;;
esac

for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
    for num_prototypes in 10 25; do
        for std in 0.1 0.25 0.5; do
            for transform_layers in 32 128-256-32; do
                for fusing_type in concat full_mult_int; do
                    # With golden proto
                    for golden_proto_weight in same 0.25 0.5 1.0; do
                        cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                            "--config.hp.head.num_prototypes $num_prototypes" \
                            "--config.hp.head.num_test_prototypes $num_prototypes" \
                            "--config.hp.head.fusing_type $fusing_type" \
                            "--config.hp.head.golden_proto.enabled true" \
                            "--config.hp.head.golden_proto.weight $golden_proto_weight" \
                            "--config.hp.head.noise.transform_layers {64-256-16}" \
                            "--config.hp.head.noise.std $std")
                        # echo "$cli_args"
                        sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                    done

                    # Without golden proto
                    cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                            "--config.hp.head.num_prototypes $num_prototypes" \
                            "--config.hp.head.num_test_prototypes $num_prototypes" \
                            "--config.hp.head.fusing_type $fusing_type" \
                            "--config.hp.head.golden_proto.enabled false" \
                            "--config.hp.head.noise.transform_layers {64-256-16}" \
                            "--config.hp.head.noise.std $std")
                    # echo "$cli_args"
                    sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                done
            done
        done
    done

    # Baseline
    cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed")
    # echo "$cli_args"
    sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
done
