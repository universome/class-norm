#!/usr/bin/env python
import sys; sys.path.append('.')
import os
import random
import argparse
from typing import Dict, List, Any, Callable

import numpy as np
from firelab.config import Config
from src.trainers.zsl_trainer import ZSLTrainer

from utils import generate_experiments_from_hpo_grid

random.seed(42)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-n', '--num_runs', type=int, help='Number of runs for each experimental setup')
    parser.add_argument('-d', '--dataset', type=str, help='Which dataset to run on?')
    parser.add_argument('-e', '--experiments_dir', type=str, help='Where to save the results?')
    parser.add_argument('--count', action='store_true', help='Should we just count and exit?')

    return parser.parse_args()


def main():
    args = read_args()
    hpos = Config.load('slurm/hpos.yml')['zsl']
    experiments_vals = generate_experiments_from_hpo_grid(hpos.grid)
    if hpos.get('search_type') == 'random':
        experiments_vals = random.sample(experiments_vals, min(len(experiments_vals), hpos.num_experiments))

    experiments_vals = [{p.replace('|', '.'): v for p, v in exp.items()} for exp in experiments_vals]
    hps = [Config(e) for e in experiments_vals]

    if args.count:
        print(f'Total number of experiments: {len(hps)} x [{args.num_runs} seed] = {len(hps) * args.num_runs}')
    else:
        run_hpo(args, hps)


def run_hpo(args, hps):
    experiments_dir = os.path.join('/ibex/scratch/skoroki/zsl-experiments', args.experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)
    default_config = Config.load('configs/zsl.yml', frozen=False)
    default_config.experiments_dir = experiments_dir

    best_last_score_hp = None
    best_best_score_hp = None
    best_last_score_val = 0
    best_best_score_val = 0

    for hp in hps:
        last_scores = []
        best_scores = []

        for random_seed in range(1, args.num_runs + 1):
            config = default_config.clone(frozen=False)
            config[args.dataset].hp = config[args.dataset].hp.overwrite(hp)
            config.random_seed = random_seed
            config.dataset = args.dataset

            trainer = ZSLTrainer(config)
            trainer.start()

            last_scores.append(trainer.curr_val_scores[2])
            best_scores.append(trainer.best_val_scores[2])

        mean_last_score = np.mean(last_scores)
        mean_best_score = np.mean(best_scores)

        if mean_last_score > best_last_score_val:
            best_last_score_val = mean_last_score
            best_last_score_hp = hp

        if mean_best_score > best_best_score_val:
            best_best_score_val = mean_best_score
            best_best_score_hp = hp

    print(f'Best last score hp (value: {best_last_score_val})')
    print(best_last_score_hp)

    print(f'Best best score hp (value: {best_best_score_val})')
    print(best_best_score_hp)


if __name__ == "__main__":
    main()
