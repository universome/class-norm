import os
import random
from time import time
from typing import Tuple, Iterable
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.losses import compute_gradient_penalty
from src.utils.lll import prune_logits
from src.models.gan import GAN
from src.models.gan_64x64 import GAN64x64
from src.trainers.task_trainer import TaskTrainer
from src.utils.plotting import plot_samples


class GenMemGANTaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        assert self.config.hp.model_type in ['genmem_gan', 'genmem_gan_64x64']

        ModelClass = GAN if self.config.hp.model_type == 'genmem_gan' else GAN64x64

        self.prev_model = ModelClass(self.config.hp.model_config).to(self.device_name)
        self.prev_model.load_state_dict(deepcopy(self.model.state_dict()))
        self.learnt_classes = np.unique(self.main_trainer.class_splits[:self.task_idx]).tolist()
        self.seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx + 1]).tolist()
        self.writer = SummaryWriter(os.path.join(self.main_trainer.paths.logs_path, f'task_{self.task_idx}'), flush_secs=5)

        if self.task_idx > 0:
            self.fixed_noise = self.main_trainer.task_trainers[self.task_idx - 1].fixed_noise
        else:
            self.fixed_noise = np.random.randn(1000, self.config.hp.model_config.z_dim).astype(np.float32)

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
            gen_start_time = time()
            self.generator_step(y)
            self.writer.add_scalar('gen/iter_time', time() - gen_start_time, self.num_iters_done)

        discr_start_time = time()
        self.discriminator_step(x, y)
        self.writer.add_scalar('discr/iter_time', time() - discr_start_time, self.num_iters_done)

    def discriminator_step(self, x: Tensor, y: Tensor):
        with torch.no_grad():
            z = self.model.generator.sample_noise(x.size(0)).to(self.device_name)
            x_fake = self.model.generator(z, y) # TODO: try picking y randomly

        logits_on_real, cls_logits_on_real = self.model.discriminator(x)
        logits_on_fake, _ = self.model.discriminator(x_fake)

        adv_loss = -logits_on_real.mean() + logits_on_fake.mean()
        grad_penalty = compute_gradient_penalty(self.model.discriminator.run_adv_head, x, x_fake)
        cls_loss = F.cross_entropy(cls_logits_on_real, y)
        discr_loss_total = adv_loss \
                           + self.config.hp.gp_coef * grad_penalty \
                           + self.config.hp.cls_loss_coef_discr * cls_loss

        self.optim['discriminator'].zero_grad()
        discr_loss_total.backward()
        self.optim['discriminator'].step()

        self.writer.add_scalar('discr/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_real', logits_on_real.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_fake', logits_on_fake.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/grad_penalty', grad_penalty.item(), self.num_iters_done)
        # self.writer.add_scalar('discr/cls_loss', cls_loss.item(), self.num_iters_done)

    def generator_step(self, y: Tensor):
        z = self.model.generator.sample_noise(y.size(0)).to(self.device_name)
        x_fake = self.model.generator(z, y)
        discr_logits_on_fake, cls_logits_on_fake = self.model.discriminator(x_fake)

        adv_loss = -discr_logits_on_fake.mean()
        cls_loss = self.criterion(prune_logits(cls_logits_on_fake, self.output_mask), y)
        distillation_loss = self.knowledge_distillation_loss()

        total_loss = adv_loss \
                        + self.config.hp.distill_loss_coef * distillation_loss \
                        + self.config.hp.cls_loss_coef_gen * cls_loss \

        self.optim['generator'].zero_grad()
        total_loss.backward()
        self.optim['generator'].step()

        self.writer.add_scalar(f'gen/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/distillation_loss', distillation_loss.item(), self.num_iters_done)

    def knowledge_distillation_loss(self) -> Tensor:
        if len(self.learnt_classes) == 0: return torch.tensor(0.)

        num_samples_per_class = self.config.hp.distill_batch_size // len(self.learnt_classes)
        y = np.tile(self.learnt_classes, num_samples_per_class)
        y = torch.tensor(y).to(self.device_name)
        z = self.model.generator.sample_noise(len(y)).to(self.device_name)
        outputs_teacher = self.prev_model.generator(z, y).view(len(y), -1)
        outputs_student = self.model.generator(z, y).view(len(y), -1)
        loss = torch.norm(outputs_teacher - outputs_student, dim=1).pow(2).mean()

        return loss / len(self.learnt_classes)

    def start(self):
        self._before_train_hook()

        assert self.is_trainable, "We do not have enough conditions to train this Task Trainer" \
                                  "(for example, previous trainers was not finished or this trainer was already run)"

        for i in tqdm(range(self.config.hp.num_iters_per_task), desc=f'Task #{self.task_idx} iteration'):
            batch = self.sample_train_batch()
            self.train_on_batch(batch)
            self.num_iters_done += 1
            self.run_after_iter_done_callbacks()

            if self.config.has('plotting') and i % self.config.plotting.samples_freq == 0:
                plot_samples(self)

        self._after_train_hook()

    def sample_train_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = min(self.config.hp.batch_size, len(self.task_ds_train))
        idx = random.sample(range(len(self.task_ds_train)), batch_size)
        x, y = zip(*[self.task_ds_train[i] for i in idx])

        return x, y
