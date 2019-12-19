import os
from typing import Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from src.utils.losses import compute_gradient_penalty
from src.utils.lll import prune_logits
from .task_trainer import TaskTrainer


class GenMemGANTaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        assert self.config.hp.model_type == 'genmem_gan'

        self.prev_model = deepcopy(self.model).to(self.device_name).eval()
        self.current_classes = self.main_trainer.class_splits[self.task_idx]
        self.previously_seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx]).tolist()
        self.seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx + 1]).tolist()
        self.writer = SummaryWriter(os.path.join(self.main_trainer.paths.logs_path, f'task_{self.task_idx}'))

    def construct_optimizer(self):
        return {
            'generator': torch.optim.Adam(self.model.generator.parameters(), **self.config.hp.gen_optim.kwargs.to_dict()),
            'discriminator': torch.optim.Adam(self.model.discriminator.parameters(), **self.config.hp.discr_optim.kwargs.to_dict())
        }

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        if self.num_iters_done % self.config.hp.num_discr_steps_per_gen_step == 0:
            self.generator_step(y)

        self.discriminator_step(x, y)

    def discriminator_step(self, x: Tensor, y: Tensor):
        with torch.no_grad():
            z = self.model.generator.sample_noise(x.size(0)).to(self.device_name)
            x_fake = self.model.generator(z, y) # TODO: try picking y randomly

        logits_on_real, cls_logits_on_real = self.model.discriminator(x)
        logits_on_fake, _ = self.model.discriminator(x_fake)

        adv_loss = -logits_on_real.mean() + logits_on_fake.mean()
        grad_penalty = compute_gradient_penalty(self.model.discriminator.run_discr_head, x, x_fake)
        cls_loss = F.cross_entropy(cls_logits_on_real, y)
        discr_loss_total = adv_loss \
                           + self.config.hp.gp_coef * grad_penalty \
                           + self.config.hp.cls_loss_coef_discr * cls_loss

        self.optim['discriminator'].zero_grad()
        discr_loss_total.backward()
        self.optim['discriminator'].step()

        self.writer.add_scalar(f'discr/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'discr/mean_pred_real', logits_on_real.mean().item(), self.num_iters_done)
        self.writer.add_scalar(f'discr/mean_pred_fake', logits_on_fake.mean().item(), self.num_iters_done)
        self.writer.add_scalar(f'discr/grad_penalty', grad_penalty.item(), self.num_iters_done)
        self.writer.add_scalar(f'discr/cls_loss', cls_loss.item(), self.num_iters_done)

    def generator_step(self, y: Tensor):
        z = self.model.generator.sample_noise(y.size(0)).to(self.device_name)
        x_fake = self.model.generator(z, y)
        discr_logits_on_fake, cls_logits_on_fake = self.model.discriminator(x_fake)

        adv_loss = -discr_logits_on_fake.mean()
        cls_loss = self.criterion(prune_logits(cls_logits_on_fake, self.output_mask), y)
        distillation_loss = self.knowledge_distillation_loss()

        total_loss = adv_loss \
                        + self.config.hp.cls_loss_coef_gen * cls_loss \
                        + self.config.hp.distill_loss_coef * distillation_loss

        self.optim['generator'].zero_grad()
        total_loss.backward()
        self.optim['generator'].step()

        self.writer.add_scalar(f'gen/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/distillation_loss', distillation_loss.item(), self.num_iters_done)

    def knowledge_distillation_loss(self):
        if len(self.previously_seen_classes) == 0: return torch.tensor(0.)

        num_prev_samples = self.config.hp.distill_batch_size // len(self.previously_seen_classes)
        y = np.tile(self.previously_seen_classes, num_prev_samples)
        y = torch.tensor(y).to(self.device_name)
        #y = np.random.choice(self.previously_seen_classes, size=self.config.hp.distill_batch_size)
        z = self.model.generator.sample_noise(len(y)).to(self.device_name)
        x_fake_teacher = self.prev_model.generator(z, y)
        x_fake_student = self.model.generator(z, y)
        loss = self.config.hp.distill_loss_coef * (x_fake_teacher - x_fake_student).norm()

        return loss
