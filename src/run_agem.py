import sys; sys.path.append('..')
from src.trainers.agem_trainer import AgemTrainer
from firelab.config import Config


def run_trainer():
    config = Config.load('../configs/agem.yml')
    config.set('experiments_dir', 'experiments')
    trainer = AgemTrainer(config)
    trainer.start()

if __name__ == '__main__':
    run_trainer()
