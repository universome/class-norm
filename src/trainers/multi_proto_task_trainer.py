import numpy as np
import torch
from torch import Tensor

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits
from src.utils.losses import compute_mean_distance


class MultiProtoTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        logits = self.model(x)

        if self.config.hp.head.aggregation_type == 'aggregate_losses':
            n_protos = logits.size(0) // y.size(0)
            batch_size = y.size(0)
            y = y.view(batch_size, 1).repeat(1, n_protos).view(batch_size * n_protos)
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)
        else:
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)

        loss = cls_loss

        # if self.config.hp.push_protos_apart.enabled:
        #     mean_distance = compute_mean_distance(protos)
        #     loss += self.config.hp.push_protos_apart.loss_coef * (-1 * mean_distance)

        #     self.writer.add_scalar('mean_distance', mean_distance.item(), self.num_iters_done)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('cls_loss', cls_loss.item(), self.num_iters_done)
