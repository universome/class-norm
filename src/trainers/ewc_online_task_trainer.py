from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from src.utils.weights_importance import compute_diagonal_fisher
from src.utils.training_utils import prune_logits
from src.trainers.task_trainer import TaskTrainer


class EWCOnlineTaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        prev_trainer = self.get_previous_trainer()

        if prev_trainer != None:
            self.weights_prev = torch.cat([p.data.view(-1) for p in self.model.parameters()])

            curr_fisher = self.compute_importances(self.train_dataloader, prev_trainer.output_mask)

            if (self.task_idx - self.config.get('start_idx', 0)) == 1:
                prev_fisher = torch.zeros_like(curr_fisher)
            else:
                prev_fisher = prev_trainer.fisher

            self.fisher = self.config.hp.fisher.gamma * prev_fisher + curr_fisher

    def is_trainable(self) -> bool:
        return (self.task_idx == 0) or (self.get_previous_trainer() != None)

    def train_on_batch(self, batch:Tuple[Tensor, Tensor]):
        self.model.train()

        loss = self.compute_loss(self.model, batch)

        if self.task_idx > 0:
            reg = self.compute_regularization()
            loss += self.config.hp.fisher.loss_coef * reg

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def get_weights_importances(self):
        return self.fisher

    def compute_importances(self, dataloader, output_mask):
        return compute_diagonal_fisher(self.model, dataloader, output_mask)

    def compute_regularization(self) -> Tensor:
        weights_curr = torch.cat([p.view(-1) for p in self.model.parameters()])
        reg = torch.dot((weights_curr - self.weights_prev).pow(2), self.get_weights_importances())

        return reg
