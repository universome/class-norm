import argparse
import sys; sys.path.append('.')
from typing import List, Dict, Any

import numpy as np
from firelab.config import Config
from firelab.utils.training_utils import fix_random_seed

from src.trainers.lll_trainer import LLLTrainer


DEFAULT_RANDOM_SEED = 1 # np.random.randint(np.iinfo(np.int32).max)


def run_trainer(args: argparse.Namespace, config_cli_args: List[str]):
    config = load_config(args, config_cli_args)
    print(config)

    for run_idx in range(args.num_runs):
        print(f'\n<======= RUN # {run_idx} =======>\n')
        fix_random_seed(config.random_seed + run_idx)
        LLLTrainer(config).start()


def load_config(args: argparse.Namespace, config_cli_args: List[str]) -> Config:
    base_config = Config.load('configs/base.yml')
    curr_config = Config.load(f'configs/{args.config_name}.yml')

    assert curr_config.has(args.dataset)

    # Setting properties from the base config
    config = base_config.all.clone()
    config = config.overwrite(base_config.get(args.dataset))

    # Setting properties from the current config
    config = config.overwrite(curr_config.all)
    config = config.overwrite(curr_config.get(args.dataset))

    # Setting experiment-specific properties
    config.set('experiments_dir', args.experiments_dir)
    config.set('random_seed', args.random_seed)

    # Overwriting with CLI arguments
    config = config.overwrite(Config.read_from_cli())

    config_cli_args_prefix = cli_config_args_to_exp_name(config_cli_args)
    exp_name = f'{args.config_name}-{args.exp_name}-{args.dataset}-{config_cli_args_prefix}-{config.random_seed}'
    config.set('exp_name', exp_name)

    return config


def cli_config_args_to_exp_name(args: List) -> str:
    # TODO: parse args in a better format... And explain these weird manipulations...
    # Explanation: check parser.parse_known_args() format
    return '_'.join([f'{args[i][len("--config."):]}={args[i + 1]}' for i in range(0, len(args) - 1, 2)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-d', '--dataset', default='cifar100', type=str, help='Dataset')
    parser.add_argument('-s', '--random_seed', type=int, default=DEFAULT_RANDOM_SEED, help='Random seed to fix')
    parser.add_argument('-c', '--config_name', type=str, default='lgm', help='Which config to run?')
    parser.add_argument('-n', '--num_runs', type=int, default=1, help='How many times we should run the experiment?')
    parser.add_argument('-e', '--exp_name', type=str, default='', help='Postfix to add to experiment name.')
    parser.add_argument('--experiments_dir', type=str, default='experiments', help='Directory where all the experiments reside.')

    args, config_args = parser.parse_known_args()

    run_trainer(args, config_args)
