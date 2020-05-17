from typing import Tuple

import torch
from torch import Tensor

from src.utils.weights_importance import compute_mse_grad
from src.trainers.ewc_online_task_trainer import EWCOnlineTaskTrainer


class MASTaskTrainer(EWCOnlineTaskTrainer):
    def _after_init_hook(self):
        prev_trainer = self.get_previous_trainer()

        if prev_trainer != None:
            self.weights_prev = torch.cat([p.data.view(-1) for p in self.model.parameters()])
            self.mse_grad = compute_mse_grad(self.model, self.train_dataloader, prev_trainer.output_mask)

    def compute_regularization(self) -> Tensor:
        weights_curr = torch.cat([p.view(-1) for p in self.model.parameters()])
        reg = torch.dot((weights_curr - self.weights_prev).pow(2), self.mse_grad)

        return reg
