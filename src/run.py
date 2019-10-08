import argparse
import sys; sys.path.append('.')

from src.trainers.lll_trainer import LLLTrainer
from firelab.config import Config
from firelab.utils.training_utils import fix_random_seed


def run_trainer(args):
    config = Config.load(f'configs/{args.config_name}.yml')
    config.set('experiments_dir', 'experiments')

    all_zst_accs = []

    for run_idx in range(args.num_runs):
        print(f'\n<======= RUN # {run_idx} =======>\n')
        fix_random_seed(config.random_seed + run_idx)
        trainer = LLLTrainer(config)
        trainer.start()
        all_zst_accs.append(trainer.zst_accs)

    print(f'<======= Zero-Shot accuracies for [{args.config_name}] =======>')

    for i, zst_accs in enumerate(all_zst_accs):
        print(f'Run #{i}', zst_accs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running LLL trainer')
    #parser.add_argument('-s', '--seed', default=42, help='Random seed to fix')
    parser.add_argument('-c', '--config_name', type=str, default='mas_cub', help='Which config to run?')
    parser.add_argument('-n', '--num_runs', type=int, default=1, help='How many times we should run the experiment?')
    args = parser.parse_args()

    run_trainer(args)
