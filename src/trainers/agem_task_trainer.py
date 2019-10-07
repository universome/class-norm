from typing import Tuple
import random

import torch
from torch import Tensor
import numpy as np

from src.trainers.task_trainer import TaskTrainer
from src.utils.lll import prune_logits
from src.utils.constants import NEG_INF


class AgemTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch:Tuple[Tensor, Tensor]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        logits = self.model(x)
        pruned_logits = prune_logits(logits, self.output_mask)
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

