import numpy as np
import torch

from src.trainers.task_trainer import TaskTrainer


class BasicTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()
        loss = self.compute_loss(self.model, batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
