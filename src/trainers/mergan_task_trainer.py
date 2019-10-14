from copy import deepcopy
from typing import Tuple

import torch
from torch import Tensor
import numpy as np

from src.trainers.task_trainer import TaskTrainer
from src.utils.lll import prune_logits
from src.utils.losses import compute_gradient_penalty
from src.utils.data_utils import compute_class_centroids


class MeRGANTaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        assert self.config.hp.model_type == 'gan_classifier'

        # TODO: replace with running centroids if you want more fair LLL setup (seeing one example only once)
        self.class_centroids = compute_class_centroids(self.task_ds_train, self.config.data.num_classes)
        self.prev_model = deepcopy(self.model).to(self.device_name).eval()
        self.current_classes = self.main_trainer.class_splits[self.task_idx]
        self.previously_seen_classes = list(set([c for task_id in range(self.task_idx) for c in self.main_trainer.class_splits[task_id]]))

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        self.discriminator_step(x, y)

        if self.num_iters_done % self.config.hp.model_config.num_discr_steps_per_gen_step == 0:
            self.generator_step(x, y)

        if self.task_idx > 0:
            self.knowledge_distillation_step()

    def discriminator_step(self, x: Tensor, y: Tensor):
        with torch.no_grad():
            z = self.model.generator.sample_noise(x.size(0)).to(self.device_name)
            x_fake = self.model.generator(z, self.model.attrs[y])

        discr_logits_on_real, cls_logits_on_real = self.model.discriminator(x)
        discr_logits_on_fake, cls_logits_on_fake = self.model.discriminator(x_fake)

        cls_pruned_logits_on_real = prune_logits(cls_logits_on_real, self.output_mask)
        cls_pruned_logits_on_fake = prune_logits(cls_logits_on_fake, self.output_mask)

        discr_loss = -discr_logits_on_real.mean() + discr_logits_on_fake.mean()
        cls_loss = self.criterion(cls_pruned_logits_on_real, y) + self.criterion(cls_pruned_logits_on_fake, y)
        grad_penalty = compute_gradient_penalty(self.model.discriminator.run_discr_head, x, x_fake)
        discr_loss_total = discr_loss + cls_loss + grad_penalty

        discr_loss_total.backward()
        self.optim['discr'].step()
        self.optim['discr'].zero_grad()

        self.writer.add_scalar(f'Train/task_{self.task_idx}/discr/discr_loss', discr_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'Train/task_{self.task_idx}/discr/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'Train/task_{self.task_idx}/discr/grad_penalty', grad_penalty.item(), self.num_iters_done)

    def generator_step(self, x: Tensor, y: Tensor):
        z = self.model.generator.sample_noise(x.size(0)).to(self.device_name)
        x_fake = self.model.generator(z, self.model.attrs[y])
        _, cls_logits_on_real = self.model.discriminator(x)
        discr_logits_on_fake, cls_logits_on_fake = self.model.discriminator(x_fake)

        discr_loss = -discr_logits_on_fake.mean()
        # TODO: CIZSL for some reason uses C_real here
        cls_loss = self.criterion(prune_logits(cls_logits_on_fake, self.output_mask), y)

        if self.config.hp.model_config.centroid_reg_coef > 0:
            centroid_loss = self.compute_centroid_loss(x_fake, y)
        else:
            centroid_loss = 0

        # TODO: CIZSL uses L2 reg manually (it's wrong to do it manually for Adam)
        # TODO: CIZSL uses additional L2 reg for generator attr embeddings

        total_loss = discr_loss + cls_loss + self.config.hp.model_config.centroid_reg_coef * centroid_loss
        total_loss.backward()
        self.optim['gen'].step()
        self.optim['gen'].zero_grad()

        self.writer.add_scalar(f'Train/task_{self.task_idx}/gen/discr_loss', discr_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'Train/task_{self.task_idx}/gen/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'Train/task_{self.task_idx}/gen/centroid_loss', centroid_loss.item(), self.num_iters_done)

    def knowledge_distillation_step(self):
        z = self.model.generator.sample_noise(self.config.hp.model_config.distill_batch_size).to(self.device_name)
        y = np.random.choice(self.previously_seen_classes, size=self.config.hp.model_config.distill_batch_size)
        x_fake_teacher = self.prev_model.generator(z, self.model.attrs[y])
        x_fake_student = self.model.generator(z, self.model.attrs[y])
        loss = self.config.hp.model_config.model_distill_coef * (x_fake_teacher - x_fake_student).norm()

        loss.backward()
        self.optim['gen'].step()
        self.optim['gen'].zero_grad()

        self.writer.add_scalar(f'Train/task_{self.task_idx}/gen/knowledge_distillation', loss.item(), self.num_iters_done)

    def compute_centroid_loss(self, x_fake, y):
        groups = {}
        for x, l in zip(x_fake, y):
            if not l in groups: groups[l] = []
            groups[l].append(x)

        centroids_fake = {l: torch.stack(groups[l]).mean(dim=0) for l in groups}
        class_centroids = torch.tensor(self.class_centroids).to(x_fake.device).float()
        distances = [(c - class_centroids[l]).norm() for l, c in centroids_fake.items()]
        loss = sum(distances) # TODO: check that this thing is differentiable

        return loss / self.config.data.num_classes
