from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
import numpy as np
from tqdm import tqdm

from src.models.classifier import ZSClassifier
from src.dataloaders.cub import load_cub_dataset, load_class_attributes
from src.utils.data_utils import (
    split_classes_for_tasks,
    get_train_test_data_splits,
    construct_output_mask,
)


NEG_INF = float('-inf')


class AgemTaskTrainer:
    def __init__(self, main_trainer:AgemTrainer, task_idx:int):
        self.task_idx = task_idx
        self.config = main_trainer.config
        self.model = main_trainer.model
        self.optim = main_trainer.optim
        self.episodic_memory = main_trainer.episodic_memory
        self.episodic_memory_output_mask = main_trainer.episodic_memory_output_mask
        self.device_name = main_trainer.device_name
        self.criterion = nn.CrossEntropyLoss()

        self.task_ds_train, self.task_ds_test = main_trainer.data_splits[task_idx]
        self.output_mask = construct_output_mask(main_trainer.class_splits[task_idx], self.config.num_classes)
        self.init_dataloaders()

    def init_dataloaders(self):
        self.train_dataloader = DataLoader(self.task_ds_train, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)))
        self.test_dataloader = DataLoader(self.task_ds_test, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)))

    def start(self):
        """Runs training"""
        for batch in tqdm(self.train_dataloader, desc=f'Task #{self.task_idx}'):
            self.train_on_batch(batch)

    def train_on_batch(self, batch:Tuple[Tensor, Tensor]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        logits = self.model(x)
        pruned_logits = self.prune_logits(logits)
        loss = self.criterion(pruned_logits, y)

        if self.config.hp.get('use_agem', True) and len(self.episodic_memory) > 0:
            ref_grad = self.compute_ref_grad()
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()])
            grad = self.project_grad(grad, ref_grad)

            self.optim.zero_grad()
            self.set_grad(grad)
        else:
            self.optim.zero_grad()
            loss.backward()

        self.optim.step()

    def compute_ref_grad(self):
        num_samples_to_use = min(self.config.hp.mem_batch_size, len(self.episodic_memory))
        batch_idx = random.sample(np.arange(len(self.episodic_memory)).tolist(), num_samples_to_use)
        batch = [m for i, m in enumerate(self.episodic_memory) if i in batch_idx]
        output_mask = np.array([m for i, m in enumerate(self.episodic_memory_output_mask) if i in batch_idx])

        x = torch.tensor([x for x,_ in batch]).to(self.device_name)
        y = torch.tensor([y for _, y in batch]).to(self.device_name)
        logits = self.model(x)
        pruned_logits = logits.masked_fill(torch.tensor(~output_mask).to(self.device_name), NEG_INF)
        loss = self.criterion(pruned_logits, y)
        loss.backward()
        ref_grad = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()])
        self.optim.zero_grad()

        return ref_grad

    def project_grad(self, grad, ref_grad):
        dot_product = torch.dot(grad, ref_grad)

        if dot_product > 0:
            return grad
        else:
            return grad - ref_grad * dot_product / ref_grad.pow(2).sum()

    def set_grad(self, grad: Tensor):
        """Takes gradient and sets it to parameters .grad"""
        assert grad.dim() == 1

        for param in self.model.parameters():
            param.grad.data = grad[:param.numel()].view(*param.shape).data
            grad = grad[param.numel():]

        assert len(grad) == 0, "Not all weights were used!"

    def prune_logits(self, logits:Tensor) -> Tensor:
        """
        Takes logits and sets those classes which do not participate
        in the current task to -infinity so they are not explicitly penalized and forgotten
        """
        mask_idx = np.nonzero(~self.output_mask)[0]
        pruned = logits.index_fill(1, torch.tensor(mask_idx).to(self.device_name), NEG_INF)

        return pruned

    def compute_accuracy(self, dataloader:DataLoader):
        guessed = []
        self.model.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x = torch.tensor(x).to(self.device_name)
                y = torch.tensor(y).to(self.device_name)

                logits = self.model(x)
                pruned_logits = self.prune_logits(logits)

                guessed.extend((pruned_logits.argmax(dim=1) == y).cpu().data.tolist())

        return np.mean(guessed)

    def compute_test_accuracy(self):
        return self.compute_accuracy(self.test_dataloader)

    def compute_train_accuracy(self):
        return self.compute_accuracy(self.train_dataloader)
