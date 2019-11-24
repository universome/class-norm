#!/usr/bin/env bash

num_runs=$1

# Running A-GEM
for (( random_seed=1; random_seed<=num_runs; random_seed++ ))
    do sbatch --export=ALL,cli_args="-c agem --random_seed $random_seed --config.hp.optim.kwargs.lr 0.005" slurm/slurm_job.sh; done

# Running MAS
for (( random_seed=1; random_seed<=num_runs; random_seed++ ))
    do sbatch --export=ALL,cli_args="-c mas --random_seed $random_seed --config.hp.synaptic_strength 0.01" slurm/slurm_job.sh; done

# Running EWC
for (( random_seed=1; random_seed<=num_runs; random_seed++ ))
    do sbatch --export=ALL,cli_args="-c ewc --random_seed $random_seed --config.hp.synaptic_strength 0.01" slurm/slurm_job.sh; done

# Running sequential
for (( random_seed=1; random_seed<=num_runs; random_seed++ ))
    do sbatch --export=ALL,cli_args="-c basic --random_seed $random_seed --config.hp.optim.kwargs.lr 0.005" slurm/slurm_job.sh; done

# Running genmem
#for (( random_seed=1; random_seed<=num_runs; random_seed++ ))
#    do sbatch --export=ALL,cli_args="-c genmem --random_seed $random_seed" slurm/slurm_job.sh; done

# Running joint model
for (( random_seed=1; random_seed<=num_runs; random_seed++ ))
    do sbatch --export=ALL,cli_args="-c joint --random_seed $random_seed" slurm/slurm_job.sh; done

# Running mergazsl
#for random_seed in {1.."$num_runs"};
# do sbatch --export=ALL,cli_args="-c mergazsl --random_seed $random_seed --config.hp.joint_cls_training_loss_coef 0.01" slurm/slurm_job.sh; done