#!/usr/bin/env python
import sys; sys.path.append('.')
import os
import argparse
from typing import Dict, List, Any, Callable

import numpy as np
from firelab.config import Config
from src.trainers.zsl_trainer import ZSLTrainer

from utils import generate_experiments_from_hpo_grid


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-n', '--num_runs', type=int, help='Number of runs for each experimental setup')
    parser.add_argument('-d', '--dataset', type=str, help='Which dataset to run on?')
    parser.add_argument('-e', '--experiment', type=str, help='Which HPO experiment to run.')

    return parser.parse_args()


def main():
    args = read_args()
    hpos = Config.load('slurm/hpos.yml')[args.experiment]
    experiments_vals = generate_experiments_from_hpo_grid(hpos.grid)

    experiments_vals = [{p.replace('|', '.'): v for p, v in exp.items()} for exp in experiments_vals]
    hps = [Config(e) for e in experiments_vals]

    run_series(args, hps)


def run_series(args, hps):
    experiments_dir = f'/ibex/scratch/skoroki/zsl-experiments/{args.experiment}'
    log_file = f'logs/{args.dataset}-{args.experiment}.log'
    os.makedirs('logs/', exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)
    default_config = Config.load('configs/zsl.yml', frozen=False)
    default_config.experiments_dir = experiments_dir

    for i, hp in enumerate(hps):
        print(f'<======= Running hp #{i+1}/{len(hps)} =======>')
        val_scores = []
        test_scores = []
        training_times = []

        for random_seed in range(1, args.num_runs + 1):
            print(f'=> Seed #{random_seed}/{args.num_runs}')
            config = default_config.clone(frozen=False)
            config[args.dataset].set('hp',config[args.dataset].hp.overwrite(hp))
            config.set('random_seed', random_seed)
            config.set('dataset', args.dataset)
            config.set('silent', True)

            trainer = ZSLTrainer(config)
            trainer.start()

            val_scores.append(trainer.curr_val_scores)
            test_scores.append(trainer.best_val_scores)
            training_times.append(trainer.elapsed)

        val_scores = np.array(val_scores)
        test_scores = np.array(test_scores)
        training_times = np.array(training_times)

        log_str = str(hp)
        log_str += f'[VAL] S: {val_scores[:,0].mean():.02f} (std: {val_scores[:,0].std():.02f}). ' \
                   f'U: {val_scores[:,1].mean():.02f} (std: {val_scores[:,1].std():.02f}). ' \
                   f'H: {val_scores[:,2].mean():.02f} (std: {val_scores[:,2].std():.02f}).' \
                   f'Z: {val_scores[:,3].mean():.02f} (std: {val_scores[:,3].std():.02f}).\n'
        log_str += f'[TEST] S: {test_scores[:,0].mean():.02f} (std: {test_scores[:,0].std():.02f}). ' \
                   f'U: {test_scores[:,1].mean():.02f} (std: {test_scores[:,1].std():.02f}). ' \
                   f'H: {test_scores[:,2].mean():.02f} (std: {test_scores[:,2].std():.02f}).' \
                   f'Z: {test_scores[:,3].mean():.02f} (std: {test_scores[:,3].std():.02f}).\n'
        log_str += f'Training time: {training_times.mean():.02f} (std: {training_times.std():.02f})'

        with open(log_file, 'a') as f:
            f.write('\n======================================\n')
            f.write(log_str)

        print(log_str)


if __name__ == "__main__":
    main()
