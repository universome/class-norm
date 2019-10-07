import sys; sys.path.append('.')

from src.trainers.lll_trainer import LLLTrainer
from firelab.config import Config
from firelab.utils.training_utils import fix_random_seed


def run_trainer():
    config = Config.load('configs/agem.yml')
    config.set('experiments_dir', 'experiments')

    all_zst_accs = []

    for run_idx in range(config.get('num_runs', 1)):
        print(f'\n\n<======= RUN # {run_idx} =======>\n\n')
        fix_random_seed(config.random_seed + run_idx)
        trainer = LLLTrainer(config)
        trainer.start()

    all_zst_accs.append(trainer.zst_accs)

    print(f'<======= Zero-Shot accuracies (A-GEM: {config.hp.use_agem}) =======>')
    for i, zst_accs in enumerate(all_zst_accs):
        print(f'Run #{i}', zst_accs)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Running AgemTrainer')
    # parser.add_argument('-s', '--seed', default=42, help='Random seed to fix')
    # args = parser.parse_args()

    run_trainer()
