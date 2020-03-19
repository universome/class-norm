### Classification degradation results
# CUB
CUDA_VISIBLE_DEVICES=0 firelab start configs/classifier.yml --tb-port 12000 --exp_dirname clf-degradation --exp_name cub_64 --config.dataset cub --config.hp.img_target_shape 64 --config.hp.model.pretrained true &
CUDA_VISIBLE_DEVICES=1 firelab start configs/classifier.yml --tb-port 12001 --exp_dirname clf-degradation --exp_name cub_128 --config.dataset cub --config.hp.img_target_shape 128 --config.hp.model.pretrained true &
CUDA_VISIBLE_DEVICES=2 firelab start configs/classifier.yml --tb-port 12002 --exp_dirname clf-degradation --exp_name cub_256 --config.dataset cub --config.hp.img_target_shape 256 --config.hp.model.pretrained true &
CUDA_VISIBLE_DEVICES=3 firelab start configs/classifier.yml --tb-port 12003 --exp_dirname clf-degradation --exp_name cub_512 --config.dataset cub --config.hp.img_target_shape 512 --config.hp.model.pretrained true --config.hp.batch_size 64 &

# TinyImageNet
CUDA_VISIBLE_DEVICES=4 firelab start configs/classifier.yml --tb-port 13000 --exp_dirname clf-degradation --exp_name tiny_imagenet_16 --config.dataset tiny_imagenet --config.hp.img_target_shape 16 &
CUDA_VISIBLE_DEVICES=5 firelab start configs/classifier.yml --tb-port 13001 --exp_dirname clf-degradation --exp_name tiny_imagenet_32 --config.dataset tiny_imagenet --config.hp.img_target_shape 32 &
CUDA_VISIBLE_DEVICES=7 firelab start configs/classifier.yml --tb-port 13002 --exp_dirname clf-degradation --exp_name tiny_imagenet_64 --config.dataset tiny_imagenet --config.hp.img_target_shape 64 &

# AwA2
CUDA_VISIBLE_DEVICES=8 firelab start configs/classifier.yml --tb-port 14000 --exp_dirname clf-degradation --exp_name awa_32 --config.dataset awa --config.hp.img_target_shape 32 &
CUDA_VISIBLE_DEVICES=0 firelab start configs/classifier.yml --tb-port 14001 --exp_dirname clf-degradation --exp_name awa_64 --config.dataset awa --config.hp.img_target_shape 64 &
CUDA_VISIBLE_DEVICES=1 firelab start configs/classifier.yml --tb-port 14002 --exp_dirname clf-degradation --exp_name awa_128 --config.dataset awa --config.hp.img_target_shape 128 &
CUDA_VISIBLE_DEVICES=2 firelab start configs/classifier.yml --tb-port 14003 --exp_dirname clf-degradation --exp_name awa_256 --config.dataset awa --config.hp.img_target_shape 256 &


### Life Long Learning results
## CUB
# DEM
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 4 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=2 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=3 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=4 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 32 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &

CUDA_VISIBLE_DEVICES=5 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=7 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 4 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=8 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=4 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 32 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &

# Simple baselines
CUDA_VISIBLE_DEVICES=1 python src/run.py -c basic -d cub --experiments_dir cub-experiments &
CUDA_VISIBLE_DEVICES=2 python src/run.py -c joint -d cub --experiments_dir cub-experiments &

# Vanilla EM
CUDA_VISIBLE_DEVICES=3 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 256 2> logs/cub-dem-256-1.log &
CUDA_VISIBLE_DEVICES=4 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 2 --config.hp.memory.downsample_size 256 2> logs/cub-dem-256-2.log &
CUDA_VISIBLE_DEVICES=5 python src/run.py -c dem -d cub --experiments_dir cub-experiments --config.hp.memory.num_samples_per_class 5 --config.hp.memory.downsample_size 256 2> logs/cub-dem-256-5.log &

# iCARL
CUDA_VISIBLE_DEVICES=7 python src/run.py -c icarl -d cub --experiments_dir cub-experiments --config.hp.memory.max_size 200 2> logs/cub_icarl_200.log &
CUDA_VISIBLE_DEVICES=8 python src/run.py -c icarl -d cub --experiments_dir cub-experiments --config.hp.memory.max_size 400 2> logs/cub_icarl_400.log &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c icarl -d cub --experiments_dir cub-experiments --config.hp.memory.max_size 1000 2> logs/cub_icarl_1000.log &

# A-GEM
CUDA_VISIBLE_DEVICES=1 python src/run.py -c agem -d cub --experiments_dir cub-experiments --config.hp.num_mem_samples_per_class 1 &
CUDA_VISIBLE_DEVICES=2 python src/run.py -c agem -d cub --experiments_dir cub-experiments --config.hp.num_mem_samples_per_class 2 &
CUDA_VISIBLE_DEVICES=3 python src/run.py -c agem -d cub --experiments_dir cub-experiments --config.hp.num_mem_samples_per_class 5 &


## Tiny ImageNet
# DEM
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 16 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-16-1.log &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 4 --config.hp.memory.downsample_size 16 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-16-4.log &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 16 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-16-8.log &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 16 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-16-16.log &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 32 --config.hp.memory.downsample_size 16 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-16-32.log &

CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-32-1.log &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 4 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-32-4.log &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-32-8.log &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-32-16.log &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 32 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/tiny_imagenet-dem-32-32.log &

# Simple baselines
CUDA_VISIBLE_DEVICES=3 python src/run.py -c basic -d tiny_imagenet --experiments_dir tiny_imagenet-experiments 2> logs/tiny_imagenet-basic.log .log &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c joint -d tiny_imagenet --experiments_dir tiny_imagenet-experiments 2> logs/tiny_imagenet-joint.log .log &

# Vanilla EM
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 64 2> logs/tiny_imagenet-dem-64-1.log &
CUDA_VISIBLE_DEVICES=3 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 2 --config.hp.memory.downsample_size 64 2> logs/tiny_imagenet-dem-64-2.log &
CUDA_VISIBLE_DEVICES=7 python src/run.py -c dem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.num_samples_per_class 5 --config.hp.memory.downsample_size 64 2> logs/tiny_imagenet-dem-64-5.log &

# iCARL
CUDA_VISIBLE_DEVICES=2 python src/run.py -c icarl -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.max_size 200 2> logs/tiny_imagenet-icarl-200.log &
CUDA_VISIBLE_DEVICES=8 python src/run.py -c icarl -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.max_size 400 2> logs/tiny_imagenet-icarl-400.log &
CUDA_VISIBLE_DEVICES=9 python src/run.py -c icarl -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.memory.max_size 1000 2> logs/tiny_imagenet-icarl-1000.log &

# A-GEM
CUDA_VISIBLE_DEVICES=4 python src/run.py -c agem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.num_mem_samples_per_class 1 2> logs/tiny_imagenet-agem-1.log &
CUDA_VISIBLE_DEVICES=5 python src/run.py -c agem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.num_mem_samples_per_class 2 2> logs/tiny_imagenet-agem-2.log &
CUDA_VISIBLE_DEVICES=6 python src/run.py -c agem -d tiny_imagenet --experiments_dir tiny_imagenet-experiments --config.hp.num_mem_samples_per_class 5 2> logs/tiny_imagenet-agem-5.log &

## AwA
# DEM
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 4 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 32 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 &

CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 4 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 32 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 &

CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 4 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 32 --config.hp.memory.downsample_size 128 --config.hp.lowres_training.logits_matching_loss_coef 1 &

# Simple baselines
CUDA_VISIBLE_DEVICES=3 python src/run.py -c basic -d awa --experiments_dir awa-experiments &
CUDA_VISIBLE_DEVICES=3 python src/run.py -c joint -d awa --experiments_dir awa-experiments &

# Vanilla EM
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 1 --config.hp.memory.downsample_size 256 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 2 --config.hp.memory.downsample_size 256 &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir awa-experiments --config.hp.memory.num_samples_per_class 5 --config.hp.memory.downsample_size 256 &

# iCARL
CUDA_VISIBLE_DEVICES=6 python src/run.py -c icarl -d awa --experiments_dir awa-experiments --config.hp.memory.max_size 200 &
CUDA_VISIBLE_DEVICES=6 python src/run.py -c icarl -d awa --experiments_dir awa-experiments --config.hp.memory.max_size 400 &
CUDA_VISIBLE_DEVICES=6 python src/run.py -c icarl -d awa --experiments_dir awa-experiments --config.hp.memory.max_size 1000 &

# A-GEM
CUDA_VISIBLE_DEVICES=7 python src/run.py -c agem -d awa --experiments_dir awa-experiments --config.hp.num_mem_samples_per_class 1 &
CUDA_VISIBLE_DEVICES=7 python src/run.py -c agem -d awa --experiments_dir awa-experiments --config.hp.num_mem_samples_per_class 2 &
CUDA_VISIBLE_DEVICES=7 python src/run.py -c agem -d awa --experiments_dir awa-experiments --config.hp.num_mem_samples_per_class 5 &


### Logits-matching ablation study
CUDA_VISIBLE_DEVICES=0 python src/run.py -c dem -d awa --experiments_dir lm-ablation-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 0 2> logs/lm-ablation-awa-lm-coef-0.log &
CUDA_VISIBLE_DEVICES=1 python src/run.py -c dem -d awa --experiments_dir lm-ablation-experiments --config.hp.memory.num_samples_per_class 16 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/lm-ablation-awa-lm-coef-1.log &
CUDA_VISIBLE_DEVICES=2 python src/run.py -c dem -d tiny_imagenet --experiments_dir lm-ablation-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 0 2> logs/lm-ablation-tiny_imagenet-lm-coef-0.log &
CUDA_VISIBLE_DEVICES=3 python src/run.py -c dem -d tiny_imagenet --experiments_dir lm-ablation-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 32 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/lm-ablation-tiny_imagenet-lm-coef-1.log &
CUDA_VISIBLE_DEVICES=4 python src/run.py -c dem -d cub --experiments_dir lm-ablation-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 0 2> logs/lm-ablation-cub-lm-coef-0.log &
CUDA_VISIBLE_DEVICES=5 python src/run.py -c dem -d cub --experiments_dir lm-ablation-experiments --config.hp.memory.num_samples_per_class 8 --config.hp.memory.downsample_size 64 --config.hp.lowres_training.logits_matching_loss_coef 1 2> logs/lm-ablation-cub-lm-coef-1.log &
