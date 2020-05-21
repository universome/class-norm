#!/usr/bin/env python

import os
import argparse
from typing import Dict, List, Any, Callable

from firelab.config import Config

from utils import generate_experiments_from_hpo_grid


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-n', '--num_runs', type=int, help='Number of runs for each experimental setup')
    parser.add_argument('-d', '--dataset', type=str, help='Which dataset to run on?')
    parser.add_argument('-c', '--config_name', type=str, help='Which config to run?')
    parser.add_argument('-e', '--experiment', type=str, help='Which HPO experiment to run.')
    parser.add_argument('--count', action='store_true', help='Flag which says that we just need to count the experiments.')
    parser.add_argument('--print', action='store_true', help='Flag which says that we just need to print the CLI arguments.')

    return parser.parse_args()


def main():
    args = read_args()
    hpos = Config.load('slurm/hpos.yml')[args.experiment]

    experiments_vals = generate_experiments_from_hpo_grid(hpos.grid)
    experiments_vals.extend([b.to_dict() for b in hpos.get('baselines', [])])
    experiments_vals = [{p.replace('|', '.'): v for p, v in exp.items()} for exp in experiments_vals]
    experiments_cli_args = [' '.join([f'--config.hp.{p} {v}' for p, v in exp.items()]) for exp in experiments_vals]

    if args.count:
        print(f'Total number of experiments: {len(experiments_cli_args)} x [{args.num_runs} seed] = {len(experiments_cli_args) * args.num_runs}')
    else:
        run_hpo(args, experiments_cli_args, print_only=args.print)


def run_hpo(args, experiments_cli_args, print_only: bool=False):
    experiments_dir = os.path.join('/ibex/scratch/skoroki/experiments', args.experiment)
    logs_dir = os.path.join('/ibex/scratch/skoroki/logs', f'{args.experiment}')

    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for random_seed in range(1, args.num_runs + 1):
        common_cli_args = f'-c {args.config_name} -d {args.dataset} --experiments_dir {experiments_dir} -s {random_seed}'

        for cli_args in experiments_cli_args:
            command = f'sbatch --account=conf-2020-neurips -o {logs_dir}/output-%j.log --mem 32G --export=ALL,cli_args="{common_cli_args} {cli_args}" slurm/slurm_lll_job.sh'
            if print_only:
                print(command)
            else:
                os.system(command)


if __name__ == "__main__":
    main()
