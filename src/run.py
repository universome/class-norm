import argparse
import sys; sys.path.append('.')
from typing import List

import numpy as np
from firelab.config import Config
from firelab.utils.training_utils import fix_random_seed

from src.trainers.lll_trainer import LLLTrainer
from src.utils.metrics import compute_average_accuracy


def run_trainer(args):
    config = Config.load(f'configs/{args.config_name}.yml')
    config.set('experiments_dir', 'experiments')
    config.set('exp_name', args.config_name)
    config.set('random_seed', args.random_seed)

    for run_idx in range(args.num_runs):
        print(f'\n<======= RUN # {run_idx} =======>\n')
        fix_random_seed(config.random_seed + run_idx)

        if config.has('live_hpo'):
            avg_accs = []

            for hpo_config in get_hpo_configs(config):
                hpo_trainer = LLLTrainer(hpo_config)
                hpo_trainer.start()

                avg_accs.append(compute_average_accuracy(hpo_trainer.accs_history))

            best_optim_kwargs = config.live_hpo.optim_kwargs_list[np.argmax(avg_accs)]
            normal_config = config.overwrite(Config({'hp': {'optim_kwargs': best_optim_kwargs}}))
            normal_config.set('num_reserved_classes', config.live_hpo.num_hpo_tasks)
            LLLTrainer(normal_config).start()
        else:
            LLLTrainer(config).start()


def get_hpo_configs(config: Config) -> List[Config]:
    return [config.overwrite(Config({'hp': {'optim_kwargs': kwg}})) for kwg in config.live_hpo.optim_kwargs_list]



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLL trainer')
    parser.add_argument('-s', '--random_seed', default=42, type=int, help='Random seed to fix')
    parser.add_argument('-c', '--config_name', type=str, default='basic_cub', help='Which config to run?')
    parser.add_argument('-n', '--num_runs', type=int, default=1, help='How many times we should run the experiment?')
    args = parser.parse_args()

    run_trainer(args)
