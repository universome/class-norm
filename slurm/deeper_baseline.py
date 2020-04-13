#!/usr/bin/env python

import os
import argparse
from typing import Dict, List, Any, Callable

from firelab.config import Config
from firelab.hpo import compute_hpo_vals_idx


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-n', '--num_runs', type=int, help='Number of runs for each experimental setup')
    parser.add_argument('-d', '--dataset', type=str, help='Which dataset to run on?')
    parser.add_argument('-c', '--config_name', type=str, help='Which config to run?')
    parser.add_argument('-e', '--experiments_dir', type=str, help='Directory where all the experiments reside.')
    parser.add_argument('--count', action='store_true', help='Flag which says that we just need to count the experiments.')
    parser.add_argument('--print', action='store_true', help='Flag which says that we just need to print the CLI arguments.')

    return parser.parse_args()


def main():
    args = read_args()

    hpo_grid = Config({
        'head|after_fuse_transform_layers': ['{512}', '{256-512}', '{512-512}', '{512-512-512}'],
        # 'head|after_fuse_non_linearity': ['none', 'relu', 'leaky_relu'],
        'head|num_prototypes': [1],
        # 'head|noise|std': [1.0],
        # 'head|tanh_on_top': [True, False]
    })

    def filter_experiments(params):
        return True
        # if params['head.after_fuse_transform_layers'] == '{512}':
        #     return params['head.after_fuse_non_linearity'] == 'none'
        # else:
        #     return params['head.after_fuse_non_linearity'] != 'none'

    experiments_vals_idx = compute_hpo_vals_idx(hpo_grid)
    experiments_vals = [{p: hpo_grid[p][i] for p, i in zip(hpo_grid.keys(), idx)} for idx in experiments_vals_idx]
    experiments_vals = [{p.replace('|', '.'): v for p, v in exp.items()} for exp in experiments_vals]
    experiments_vals = list(filter(filter_experiments, experiments_vals))

    # Add the baseline
    # experiments_vals.append({'head.after_fuse_transform_layers': '{512}'})
    # experiments_vals.append({'head.after_fuse_transform_layers': '{256-512}'})
    # experiments_vals.append({'head.after_fuse_transform_layers': '{256-512}'})

    experiments_cli_args = [' '.join([f'--config.hp.{p} {v}' for p, v in exp.items()]) for exp in experiments_vals]

    if args.count:
        print(f'Total number of experiments: {len(experiments_cli_args)} x [{args.num_runs} seed]')
    elif args.print:
        print('\n'.join(experiments_cli_args))
    else:
        run_hpo(args, experiments_cli_args)


def run_hpo(args, experiments_cli_args):
    os.makedirs(args.experiments_dir, exist_ok=True)

    for random_seed in range(1, args.num_runs + 1):
        common_cli_args = f'-c {args.config_name} -d {args.dataset} --experiments_dir {args.experiments_dir} -s {random_seed}'

        for cli_args in experiments_cli_args:
            if args.dataset == 'cub':
                mem = '64G'
            elif args.dataset == 'awa':
                mem = '256G'
            else:
                mem = '128G'

            command = f'sbatch --mem {mem} --export=ALL,cli_args="{common_cli_args} {cli_args}" slurm/slurm_lll_job.sh'
            # command = f'echo "sbatch --mem {mem} --export=ALL,cli_args=\"{common_cli_args} {cli_args}\" slurm/slurm_lll_job.sh"'
            os.system(command)


if __name__ == "__main__":
    main()
