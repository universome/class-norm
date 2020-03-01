import numpy as np
import torch
import torch.nn.functional as F

from src.trainers.task_trainer import TaskTrainer
from src.utils.lll import prune_logits
from src.utils.training_utils import compute_accuracy


class EMTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        logits = self.model(x)
        pruned_logits = prune_logits(logits, self.output_mask)

        cls_loss = F.cross_entropy(pruned_logits, y)
        cls_acc = compute_accuracy(pruned_logits, y)

        total_loss = cls_loss

        if self.config.hp.lowres_training.enabled or self.config.hp.lowres_training.distill:
            x_lowres = self.transform_em_sample(x)
            logits_lowres = self.model(x_lowres)
            pruned_logits_lowres = prune_logits(logits_lowres, self.output_mask)
            cls_loss_lowres = F.cross_entropy(pruned_logits_lowres, y)
            cls_acc_lowres = compute_accuracy(pruned_logits_lowres, y)

            self.writer.add_scalar('train/cls_loss_lowres', cls_loss_lowres.item(), self.num_iters_done)
            self.writer.add_scalar('train/cls_acc_lowres', cls_acc_lowres.item(), self.num_iters_done)

        if self.config.hp.lowres_training.enabled:
            total_loss += self.config.hp.lowres_training.loss_coef * cls_loss

        if self.config.hp.lowres_training.distill:
            distill_lowres_loss = F.mse_loss(logits, logits_lowres)
            total_loss += self.config.hp.lowres_training.distill_coef * distill_lowres_loss

            self.writer.add_scalar('train/distill_lowres_loss', distill_lowres_loss.item(), self.num_iters_done)

        if self.task_idx > 0:
            rehearsal_loss, rehearsal_acc = self.compute_rehearsal_loss()
            total_loss += self.config.hp.memory.loss_coef * rehearsal_loss

            self.writer.add_scalar('train/rehearsal_loss', rehearsal_loss.item(), self.num_iters_done)
            self.writer.add_scalar('train/rehearsal_acc', rehearsal_acc.item(), self.num_iters_done)


        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        self.writer.add_scalar('train/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar('train/cls_acc', cls_acc.item(), self.num_iters_done)

    def transform_em_sample(self, x):
        assert x.ndim == 4
        return F.interpolate(x, size=self.config.hp.memory.downsample_size)

    def compute_rehearsal_loss(self):
        x, y = self.sample_from_memory(self.config.hp.memory.batch_size)
        x = self.transform_em_sample(x)
        pruned_logits = prune_logits(self.model(x), self.learned_classes_mask)
        cls_loss = F.cross_entropy(pruned_logits, y)
        cls_acc = compute_accuracy(pruned_logits, y)

        return cls_loss, cls_acc

    def extend_episodic_memory(self):
        self.episodic_memory.extend(self.task_ds_train)
