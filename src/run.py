import argparse
import sys; sys.path.append('.')
from typing import List, Dict, Any

import numpy as np
from firelab.config import Config
from firelab.utils.training_utils import fix_random_seed

from src.trainers.lll_trainer import LLLTrainer


CONFIG_ARG_PREFIX = '--config.'
DEFAULT_RANDOM_SEED = np.random.randint(np.iinfo(np.int32).max)


def run_trainer(args: argparse.Namespace, config_cli_args: List[str]):
    config = load_config(args, config_cli_args)
    print(config)

    for run_idx in range(args.num_runs):
        print(f'\n<======= RUN # {run_idx} =======>\n')
        fix_random_seed(config.random_seed + run_idx)
        LLLTrainer(config).start()


def load_config(args: argparse.Namespace, config_cli_args: List[str]) -> Config:
    base_config = Config.load('configs/base.yml')
    config = Config.load(f'configs/{args.config_name}.yml')

    # Setting experiment-specific properties
    config.set('experiments_dir', 'experiments')
    config.set('random_seed', args.random_seed)
    config.set('exp_name', f'{args.config_name}-{config.random_seed}')

    # Setting properties from the base config
    config.set('data', base_config.datasets.get(args.dataset))
    config = base_config.base.overwrite(config)
    hp = config.hp.overwrite(config.hp_for_datasets.get(args.dataset))
    config = config.overwrite(Config({'hp': hp.to_dict()}))

    # Overwriting with CLI arguments
    config_cli_args: Dict = process_cli_config_args(config_cli_args)
    config = config.overwrite(Config(config_cli_args))

    return config


def process_cli_config_args(config_args:List[str]) -> Dict:
    """Takes config args from the CLI and converts them to a dict"""
    # assert len(config_args) % 3 == 0, \
    #     "You should pass config args in [--config.arg_name arg_value arg_type] format"
    assert len(config_args) % 2 == 0, \
        "You should pass config args in [--config.arg_name arg_value] format"
    arg_names = [config_args[i] for i in range(0, len(config_args), 2)]
    arg_values = [config_args[i] for i in range(1, len(config_args), 2)]

    result = {}

    for name, value in zip(arg_names, arg_values):
        assert name.startswith(CONFIG_ARG_PREFIX), \
            f"Argument {name} is unkown and does not start with `config.` prefix. Cannot parse it."

        result[name[len(CONFIG_ARG_PREFIX):]] = infer_type_and_convert(value)

    return result


def infer_type_and_convert(value:str) -> Any:
    """
    Chances are high that this function should never exist...
    It tries to get a proper type and converts the value to it.
    """
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.isdigit():
        return int(value)
    elif is_float(value):
        return float(value)
    else:
        return value


def is_float(value:Any) -> bool:
    """One more dirty function: it checks if the string is float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_hpo_configs(config: Config) -> List[Config]:
    return [config.overwrite(Config({'hp': {'optim_kwargs': kwg}})) for kwg in config.live_hpo.optim_kwargs_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-d', '--dataset', default='cub', type=str, help='Dataset')
    parser.add_argument('-s', '--random_seed', type=int, default=DEFAULT_RANDOM_SEED, help='Random seed to fix')
    parser.add_argument('-c', '--config_name', type=str, default='genmem', help='Which config to run?')
    parser.add_argument('-n', '--num_runs', type=int, default=1, help='How many times we should run the experiment?')

    args, config_args = parser.parse_known_args()

    run_trainer(args, config_args)
