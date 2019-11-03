#!/usr/bin/env bash

# Running A-GEM
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c agem --random_seed $random_seed" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c agem --random_seed $random_seed --config.hp.optim_kwargs.lr 0.005" slurm/slurm_job.sh; done

# Running MAS
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mas --random_seed $random_seed --config.hp.synaptic_strength 0.001" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mas --random_seed $random_seed --config.hp.synaptic_strength 0.01" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mas --random_seed $random_seed --config.hp.synaptic_strength 0.1" slurm/slurm_job.sh; done

# Running EWC
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c ewc --random_seed $random_seed --config.hp.synaptic_strength 0.001" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c ewc --random_seed $random_seed --config.hp.synaptic_strength 0.01" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c ewc --random_seed $random_seed --config.hp.synaptic_strength 0.1" slurm/slurm_job.sh; done

# Running basic
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c basic --random_seed $random_seed" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c basic --random_seed $random_seed --config.hp.optim_kwargs.lr 0.005" slurm/slurm_job.sh; done

# Running mergazsl
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mergazsl --random_seed $random_seed" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mergazsl --random_seed $random_seed --config.hp.joint_cls_training_loss_coef 0.1" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mergazsl --random_seed $random_seed --config.hp.joint_cls_training_loss_coef 10" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mergazsl --random_seed $random_seed --config.hp.joint_cls_training_loss_coef 0.01" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mergazsl --random_seed $random_seed --config.hp.model_config.model_distill_coef 0.01" slurm/slurm_job.sh; done
for random_seed in {1..5};
    do sbatch --export=ALL,cli_args="-c mergazsl --random_seed $random_seed --config.hp.model_config.model_distill_coef 0.0001" slurm/slurm_job.sh; done
