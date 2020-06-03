import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch import autograd

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits

class MultiProtoTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        logits = prune_logits(self.model(x), self.output_mask)
        loss = self.criterion(logits, y)

        self.optim.zero_grad()
        loss.backward()
        if self.config.hp.get('clip_grad.value', float('inf')) < float('inf'):
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.hp.clip_grad.value)
            self.writer.add_scalar('cls/grad_norm', grad_norm, self.num_iters_done)
        self.optim.step()

        self.writer.add_scalar('loss', loss.item(), self.num_iters_done)
