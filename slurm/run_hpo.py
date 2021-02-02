#!/usr/bin/env python

import os
import random
import argparse
from typing import Dict, List, Any, Callable

from firelab.config import Config

from utils import generate_experiments_from_hpo_grid, SBATCH_ARGS_STR, get_git_hash, create_project_dir, convert_config_to_cli_args

random.seed(42)

DATASET_FULL_NAME = {
    'cub': 'CUB_200_2011',
    'awa': 'Animals_with_Attributes2',
    'sun': 'SUN'
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-n', '--num_runs', type=int, help='Number of runs for each experimental setup')
    parser.add_argument('-d', '--dataset', type=str, help='Which dataset to run on?')
    # parser.add_argument('-c', '--config_name', type=str, help='Which config to run?')
    parser.add_argument('-y', '--force_old_exp_delete', action='store_true', help='Should we force delete old exp if needed?')
    parser.add_argument('-e', '--experiment', type=str, help='Which HPO experiment to run.')
    parser.add_argument('-r', '--runner', default='lll', type=str, help='Which runner to use: lll or firelab.')
    parser.add_argument('--pos_account', action='store_true', help='Should we run in the priveleged queue?')
    parser.add_argument('--count', action='store_true', help='Flag which says that we just need to count the experiments.')
    parser.add_argument('--print', action='store_true', help='Flag which says that we just need to print the CLI arguments.')

    return parser.parse_args()


def main():
    args = read_args()
    hpos = Config.load('configs/experiments.yml')[args.experiment]
    hpo_dicts = generate_experiments_from_hpo_grid(hpos.grid)
    hpo_configs = [Config(d) for d in hpo_dicts]

    if hpos.get('search_type') == 'random':
        hpo_configs = random.sample(hpo_configs, min(len(hpo_configs), hpos.num_experiments))

    hpo_configs.extend([b for b in hpos.get('baselines', [])])
    # experiments_vals = [{p.replace('|', '.'): v for p, v in exp.items()} for exp in experiments_vals]
    # experiments_cli_args = [' '.join([f'--config.hp.{p} {v}' for p, v in exp.items()]) for exp in experiments_vals]

    if args.count:
        print(f'Total number of experiments: {len(hpo_configs)} x [{args.num_runs} seed] = {len(hpo_configs) * args.num_runs}')
    else:
        run_hpo(args, hpo_configs, hpos.config_names, print_only=args.print)


def run_hpo(args, hpo_configs, config_names, print_only: bool=False):
    project_dir = f'/ibex/scratch/skoroki/czsl/experiments/{args.experiment}-{args.dataset}-{get_git_hash()}'
    logs_args_str = f'-o {project_dir}/logs/%j.log -e {project_dir}/logs/%j.err'
    runner = 'slurm_lll_job.sh' if args.runner == 'lll' else 'slurm_firelab_job.sh'
    account = "-A conf-gpu-2020.11.23" if args.pos_account else ""

    is_project_created = create_project_dir(project_dir, force_delete=args.force_old_exp_delete)
    assert is_project_created
    os.makedirs(f'{project_dir}/runs')
    os.makedirs(f'{project_dir}/logs')

    for config_name in config_names:
        for random_seed in range(1, args.num_runs + 1):
            experiment_dir = f'{project_dir}/runs/{config_name}-{random_seed}'
            common_cli_args = f'-c {config_name} -d {args.dataset} --experiment_dir {experiment_dir} -s {random_seed}'

            for config in hpo_configs:
                cli_args = convert_config_to_cli_args(config)
                # experiments_vals = [{p.replace('|', '.'): v for p, v in exp.items()} for exp in experiments_vals]
                # experiments_cli_args = [' '.join([f'--config.hp.{p} {v}' for p, v in exp.items()]) for exp in experiments_vals]
                export = f'ALL,cli_args="{common_cli_args} {cli_args}",project_dir={project_dir},dataset_full_name={DATASET_FULL_NAME.get(args.dataset)},dataset={args.dataset}'
                command = f'sbatch {SBATCH_ARGS_STR} {account} {logs_args_str} --export={export} {project_dir}/slurm/{runner}'

                if print_only:
                    print(command)
                else:
                    os.system(command)


if __name__ == "__main__":
    main()
