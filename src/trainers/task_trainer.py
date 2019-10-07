import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from src.utils.data_utils import construct_output_mask
from src.utils.lll import prune_logits


class TaskTrainer:
    def __init__(self, main_trainer: "LLLTrainer", task_idx:int):
        self.task_idx = task_idx
        self.main_trainer = main_trainer
        self.config = main_trainer.config
        self.model = main_trainer.model
        self.optim = main_trainer.optim
        self.device_name = main_trainer.device_name
        self.criterion = nn.CrossEntropyLoss()

        self.task_ds_train, self.task_ds_test = main_trainer.data_splits[task_idx]
        self.output_mask = construct_output_mask(main_trainer.class_splits[task_idx], self.config.data.num_classes)
        self.init_dataloaders()

        self._after_init_hook()

    def init_dataloaders(self):
        self.train_dataloader = DataLoader(self.task_ds_train, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)))
        self.test_dataloader = DataLoader(self.task_ds_test, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)))

    def _after_init_hook(self):
        pass

    def start(self):
        """Runs training"""
        for batch in tqdm(self.train_dataloader, desc=f'Task #{self.task_idx}'):
            self.train_on_batch(batch)

    def train_on_batch(self, batch):
        raise NotImplementedError


    def compute_accuracy(self, dataloader:DataLoader):
        guessed = []
        self.model.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x = torch.tensor(x).to(self.device_name)
                y = torch.tensor(y).to(self.device_name)

                logits = self.model(x)
                pruned_logits = prune_logits(logits, self.output_mask)

                guessed.extend((pruned_logits.argmax(dim=1) == y).cpu().data.tolist())

        return np.mean(guessed)

    def compute_test_accuracy(self):
        return self.compute_accuracy(self.test_dataloader)

    def compute_train_accuracy(self):
        return self.compute_accuracy(self.train_dataloader)
