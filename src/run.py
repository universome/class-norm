import sys; sys.path.append('.')
import argparse
from hashlib import sha256
from typing import List, Dict, Any

import numpy as np
from firelab.config import Config
from firelab.utils.training_utils import fix_random_seed

from src.trainers.lll_trainer import LLLTrainer
from src.utils.constants import DEBUG

DEFAULT_RANDOM_SEED = 1 # np.random.randint(np.iinfo(np.int32).max)


def run_trainer(args: argparse.Namespace, config_cli_args: List[str]):
    config = load_config(args, config_cli_args)
    print(config)

    fix_random_seed(config.random_seed, enable_cudnn_deterministic=True, disable_cudnn_benchmark=True)
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

    hp_hash = sha256(str(config.hp).encode('utf-8')).hexdigest()[:10]
    exp_name = f'{args.config_name}-{args.dataset}-{hp_hash}-{config.random_seed}'
    config.set('exp_name', exp_name)

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-s', '--random_seed', type=int, default=DEFAULT_RANDOM_SEED, help='Random seed to fix')
    parser.add_argument('-c', '--config_name', type=str, help='Which config to run?')
    parser.add_argument('--experiments_dir', type=str,
        default=(f'experiments{"-debug" if DEBUG else ""}'),
        help='Directory where all the experiments reside.')

    args, config_args = parser.parse_known_args()

    print('ARGS:', args)
    print('CONFIG_ARGS:', config_args)

    run_trainer(args, config_args)
