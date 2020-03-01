#!/usr/bin/env bash

num_runs=$1

# # Checking how creativity loss is working
# for creativity_loss_coef in 0 0.001 0.01 0.1 1 10; do
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         cli_args="-c lgm -d cub -s $random_seed\
#             --config.hp.num_iters_per_task 250\
#             --config.hp.creativity.enabled true\
#             --config.hp.creativity.adv_coef $creativity_loss_coef\
#             --config.hp.creativity.entropy_coef $creativity_loss_coef"
#         # echo "cli_args: $cli_args"
#         sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#     done
# done

# # Running lgm on CUB
# for synaptic_strength in 0.1 0.01; do
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         cli_args="-c lgm -d cub -s $random_seed\
#             --config.hp.num_iters_per_task 250\
#             --config.hp.creativity.enabled false\
#             --config.hp.reg_strategy ewc\
#             --config.hp.synaptic_strength $synaptic_strength"
#         # echo "cli_args: $cli_args"
#         sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#     done
# done

# for synaptic_strength in 0.1 0.01 0.001; do
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         cli_args="-c lgm -d cub -s $random_seed\
#             --config.hp.num_iters_per_task 250\
#             --config.hp.creativity.enabled false\
#             --config.hp.reg_strategy mas\
#             --config.hp.synaptic_strength $synaptic_strength"
#         # echo "cli_args: $cli_args"
#         sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#     done
# done

# # Running lgm on AWA
# for synaptic_strength in 0.1 0.01; do
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         cli_args="-c lgm -d awa -s $random_seed\
#             --config.hp.num_iters_per_task 5000\
#             --config.hp.creativity.enabled false\
#             --config.hp.reg_strategy ewc\
#             --config.hp.synaptic_strength $synaptic_strength"
#         # echo "cli_args: $cli_args"
#         sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#     done
# done

# for synaptic_strength in 0.1 0.01 0.001; do
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         cli_args="-c lgm -d awa -s $random_seed\
#             --config.hp.num_iters_per_task 5000\
#             --config.hp.creativity.enabled false\
#             --config.hp.reg_strategy mas\
#             --config.hp.synaptic_strength $synaptic_strength"
#         # echo "cli_args: $cli_args"
#         sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#     done
# done

# Running Latent GM
# for num_iters in 100 500 2500; do
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         cli_args="-c lgm -d cub -s $random_seed\
#             --config.hp.num_iters_per_task $num_iters"
#         sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#     done
# done


# Running other methods
# for dataset in cub awa; do
#     # Running A-GEM
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         sbatch --export=ALL,cli_args="-c agem -d $dataset -s $random_seed" slurm/slurm_lll_job.sh;
#     done

#     # Running MAS
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         sbatch --export=ALL,cli_args="-c mas -d $dataset -s $random_seed --config.hp.synaptic_strength 0.01" slurm/slurm_lll_job.sh;
#     done

#     # Running EWC
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         sbatch --export=ALL,cli_args="-c ewc -d $dataset -s $random_seed --config.hp.synaptic_strength 0.01" slurm/slurm_lll_job.sh;
#     done

#     # Running sequential
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         sbatch --export=ALL,cli_args="-c basic -d $dataset -s $random_seed" slurm/slurm_lll_job.sh;
#     done

#     # Running joint model
#     for (( random_seed=1; random_seed<=num_runs; random_seed++ )); do
#         sbatch --export=ALL,cli_args="-c joint -d $dataset -s $random_seed" slurm/slurm_lll_job.sh;
#     done
# done

# Running classifier with different resolutions
for dataset in cub awa; do
    for img_size in 448 256 200 128 64; do
        cli_args="--config.hp.img_target_shape $img_size\
                  --config.dataset $dataset\
                  --exp_name classifier_${dataset}_${img_size}"
        sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_firelab_job.sh;
        # echo $cli_args
    done
done

# for distill_loss_coef in 0 1; do
#     for em_resolution in 228 128 64; do
#         for dataset in cub awa; do
#             for loss_coef in 0 0.1 1 10; do
#                 cli_args="--config.hp.memory.downsample_size $em_resolution\
#                           --config.hp.lowres_training.loss_coef $loss_coef\
#                           --config.hp.lowres_training.distill_loss_coef $distill_loss_coef\
#                           -d $dataset"
#                 # sbatch --export=ALL,cli_args="$cli_args" slurm/slurm_lll_job.sh;
#                 echo $cli_args
#             done
#         done
#     done
# done
