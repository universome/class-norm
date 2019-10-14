import torch

from src.trainers.task_trainer import TaskTrainer
from src.utils.lll import prune_logits


class BasicTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        pruned_logits = self.model.compute_pruned_predictions(x, self.output_mask)
        loss = self.criterion(pruned_logits, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()