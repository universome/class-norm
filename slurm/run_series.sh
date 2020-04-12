#!/usr/bin/env bash

num_runs=$1

experiments_dir="generative_training"
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
        for generative_training_type in gdpp mmd; do
            for generative_training_loss_coef in 0.1 0.5 1.0; do
                for num_generative_protos in 5 10 25; do
                    cli_args=$(echo "-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed" \
                            "--config.hp.generative_training.enabled true" \
                            "--config.hp.generative_training.type $generative_training_type" \
                            "--config.hp.generative_training.loss_coef $generative_training_loss_coef" \
                            "--config.hp.generative_training.num_protos $num_generative_protos" \
                            "--config.hp.head.num_prototypes $num_prototypes")
                    # echo "$cli_args"
                    sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
                done
            done
        done
    done

    # Baseline
    cli_args="-c $config -d $dataset --experiments_dir $experiments_dir -s $random_seed"
    # echo "$cli_args"
    sbatch --mem "$mem" --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
done
