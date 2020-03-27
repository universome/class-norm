import numpy as np
import torch

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits
from src.utils.losses import compute_mean_distance


class MultiProtoTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        logits, protos = self.model(x, return_protos=True)
        cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)
        loss = cls_loss

        if self.config.hp.push_protos_apart.enabled:
            mean_distance = compute_mean_distance(protos)
            loss += self.config.hp.push_protos_apart.loss_coef * (-1 * mean_distance)

            self.writer.add_scalar('mean_distance', mean_distance.item(), self.num_iters_done)


        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('cls_loss', cls_loss.item(), self.num_iters_done)
