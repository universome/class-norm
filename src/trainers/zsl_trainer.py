from firelab.base_trainer import BaseTrainer
from firelab.config import Config

class ZSLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)

    def init_dataloaders(self):
        pass

    def train_on_batch(self):
        pass

    def init_models(self):
        self.model = nn.Sequential(
            self.Linear(self.config)
        )