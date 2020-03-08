import numpy as np
import torch
import torch.nn.functional as F

from src.trainers.task_trainer import TaskTrainer
from src.utils.lll import prune_logits
from src.utils.training_utils import compute_accuracy, construct_optimizer
from src.models.upsampler import Upsampler


class DEMTaskTrainer(TaskTrainer):
    def init_models(self):
        self.model = self.main_trainer.model

        if self.task_idx == 0:
            self.upsampler = Upsampler(self.config).to(self.device_name)
        else:
            self.upsampler = self.get_previous_trainer().upsampler

    def construct_optimizer(self):
        if self.config.hp.upsampler.mode == 'learnable':
            parameters = list(self.model.parameters()) + list(self.upsampler.parameters())
        else:
            parameters = self.model.parameters()

        return construct_optimizer(parameters, self.config.hp.optim)

    def train_on_batch(self, batch):
        self.model.train()

        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        logits = self.model(x)
        pruned_logits = prune_logits(logits, self.output_mask)

        cls_loss = F.cross_entropy(pruned_logits, y)
        cls_acc = compute_accuracy(pruned_logits, y)

        total_loss = cls_loss

        if self.config.hp.lowres_training.loss_coef > 0 or self.config.hp.lowres_training.logits_matching_loss_coef > 0:
            x_lowres = self.transform_em_sample(x, no_grad=False)
            logits_lowres = self.model(x_lowres)
            pruned_logits_lowres = prune_logits(logits_lowres, self.output_mask)
            cls_loss_lowres = F.cross_entropy(pruned_logits_lowres, y)
            cls_acc_lowres = compute_accuracy(pruned_logits_lowres, y)

            self.writer.add_scalar('train/cls_loss_lowres', cls_loss_lowres.item(), self.num_iters_done)
            self.writer.add_scalar('train/cls_acc_lowres', cls_acc_lowres.item(), self.num_iters_done)

        if self.config.hp.lowres_training.loss_coef > 0:
            total_loss += self.config.hp.lowres_training.loss_coef * cls_loss

        if self.config.hp.lowres_training.logits_matching_loss_coef > 0:
            logits_matching_loss = F.mse_loss(logits, logits_lowres)
            total_loss += self.config.hp.lowres_training.logits_matching_loss_coef * logits_matching_loss

            self.writer.add_scalar('train/logits_matching_loss', logits_matching_loss.item(), self.num_iters_done)

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

    def transform_em_sample(self, x, no_grad=False):
        assert x.ndim == 4
        downsampled = F.interpolate(x, size=self.config.hp.memory.downsample_size)

        if no_grad:
            with torch.no_grad():
                upsampled = self.upsampler(downsampled)
        else:
            upsampled = self.upsampler(downsampled)

        return upsampled

    def compute_rehearsal_loss(self):
        x, y = self.sample_from_memory(self.config.hp.memory.batch_size)
        x = self.transform_em_sample(x, no_grad=True)
        pruned_logits = prune_logits(self.model(x), self.learned_classes_mask)
        cls_loss = F.cross_entropy(pruned_logits, y)
        cls_acc = compute_accuracy(pruned_logits, y)

        return cls_loss, cls_acc

    def extend_episodic_memory(self):
        self.episodic_memory.extend(self.task_ds_train)
