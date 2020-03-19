import os
from copy import deepcopy
from typing import Tuple

import torch
from torch import Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits
from src.utils.losses import compute_gradient_penalty
from src.utils.data_utils import compute_class_centroids, flatten


class MeRGAZSLTaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        assert self.config.hp.model.type == 'feat_gan_classifier'

        # TODO: replace with running centroids if you want more fair LLL setup (seeing one example only once)
        self.class_centroids = compute_class_centroids(self.task_ds_train, self.config.lll_setup.num_classes)
        self.prev_model = deepcopy(self.model).to(self.device_name).eval()
        self.current_classes = self.main_trainer.class_splits[self.task_idx]
        self.learnt_classes = np.unique(flatten(self.main_trainer.class_splits[:self.task_idx])).tolist()
        self.seen_classes = np.unique(flatten(self.main_trainer.class_splits[:self.task_idx + 1])).tolist()
        self.writer = SummaryWriter(os.path.join(self.main_trainer.paths.logs_path, f'task_{self.task_idx}'))

        self.optim = {
            'gen': torch.optim.Adam(self.model.generator.parameters(),
                                   **self.config.hp.model.gen_optim.kwargs.to_dict()),
            'discr': torch.optim.Adam(self.model.discriminator.parameters(),
                                     **self.config.hp.model.discr_optim.kwargs.to_dict()),
            'cls': torch.optim.Adam(self.model.classifier.parameters(),
                                   **self.config.hp.model.cls_optim.kwargs.to_dict()),
        }

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        self.discriminator_step(x, y)
        self.classifier_step(x, y)

        if self.num_iters_done % self.config.hp.model.num_discr_steps_per_gen_step == 0:
            self.generator_step(y)

        if self.task_idx > 0:
            self.knowledge_distillation_step()

        if self.task_idx > 0 and self.config.hp.get('use_joint_cls_training'):
            self.classifier_trainer_step()

    def discriminator_step(self, x: Tensor, y: Tensor):
        with torch.no_grad():
            z = self.model.generator.sample_noise(x.size(0)).to(self.device_name)
            x_fake = self.model.generator(z, self.model.attrs[y])

        logits_on_real = self.model.discriminator(x)
        logits_on_fake = self.model.discriminator(x_fake)

        discr_loss = -logits_on_real.mean() + logits_on_fake.mean()
        grad_penalty = compute_gradient_penalty(self.model.discriminator, x, x_fake)
        discr_loss_total = discr_loss + grad_penalty

        self.optim['discr'].zero_grad()
        discr_loss_total.backward()
        self.optim['discr'].step()

        self.writer.add_scalar(f'discr/discr_loss', discr_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'discr/grad_penalty', grad_penalty.item(), self.num_iters_done)

    def classifier_step(self, x: Tensor, y: Tensor):
        pruned_logits = self.model.classifier.compute_pruned_predictions(x, self.output_mask)
        loss = self.criterion(pruned_logits, y)
        acc = (pruned_logits.argmax(axis=1) == y).float().mean()

        self.optim['cls'].zero_grad()
        loss.backward()
        if self.config.hp.has('clip_grad_norm'):
            grad_norm = clip_grad_norm_(self.model.classifier.parameters(), self.config.hp.clip_grad_norm)
            self.writer.add_scalar(f'cls/grad_norm', grad_norm, self.num_iters_done)
        self.optim['cls'].step()

        self.writer.add_scalar(f'cls/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'cls/acc', acc.item(), self.num_iters_done)

    def generator_step(self, y: Tensor):
        z = self.model.generator.sample_noise(y.size(0)).to(self.device_name)
        x_fake = self.model.generator(z, self.model.attrs[y])
        discr_logits_on_fake = self.model.discriminator(x_fake)
        cls_logits_on_fake = self.model.classifier(x_fake)

        discr_loss = -discr_logits_on_fake.mean()
        # TODO: CIZSL for some reason additionally uses C_real here. Why?
        cls_loss = self.criterion(prune_logits(cls_logits_on_fake, self.output_mask), y)

        if self.config.hp.model.centroid_reg_coef > 0:
            centroid_loss = self.compute_centroid_loss(x_fake, y)
        else:
            centroid_loss = 0

        # TODO: CIZSL uses L2 reg manually (it's wrong to do it manually for Adam)
        # TODO: CIZSL uses additional L2 reg for generator attr embeddings

        total_loss = discr_loss + cls_loss + self.config.hp.model.centroid_reg_coef * centroid_loss

        self.optim['gen'].zero_grad()
        total_loss.backward()
        self.optim['gen'].step()

        self.writer.add_scalar(f'gen/discr_loss', discr_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/centroid_loss', centroid_loss.item(), self.num_iters_done)

    def knowledge_distillation_step(self):
        z = self.model.generator.sample_noise(self.config.hp.model.distill_batch_size).to(self.device_name)
        y = np.random.choice(self.learnt_classes, size=self.config.hp.model.distill_batch_size)
        x_fake_teacher = self.prev_model.generator(z, self.model.attrs[y])
        x_fake_student = self.model.generator(z, self.model.attrs[y])
        loss = self.config.hp.model.model_distill_coef * (x_fake_teacher - x_fake_student).norm()

        self.optim['gen'].zero_grad()
        loss.backward()
        self.optim['gen'].step()

        self.writer.add_scalar(f'gen/distillation_loss', loss.item(), self.num_iters_done)

    def classifier_trainer_step(self):
        # Randomly sampling classes
        y = np.random.choice(self.seen_classes, size=self.config.hp.joint_clf_training_batch_size)
        y = torch.tensor(y).to(self.device_name).long()

        with torch.no_grad():
            z = self.model.generator.sample_noise(y.size(0)).to(self.device_name)
            x = self.model.generator(z, self.model.attrs[y])

        seen_classes_output_mask = np.zeros(self.config.lll_setup.num_classes).astype(bool)
        seen_classes_output_mask[self.seen_classes] = True
        pruned_logits = self.model.classifier.compute_pruned_predictions(x, seen_classes_output_mask)
        loss = self.criterion(pruned_logits, y)
        loss *= self.config.hp.joint_cls_training_loss_coef
        acc = (pruned_logits.argmax(axis=1) == y).float().mean()

        self.optim['cls'].zero_grad()
        loss.backward()
        if self.config.hp.has('clip_grad_norm'):
            grad_norm = clip_grad_norm_(self.model.classifier.parameters(), self.config.hp.clip_grad_norm)
            self.writer.add_scalar(f'cls/grad_norm_on_synthetic', grad_norm, self.num_iters_done)
        self.optim['cls'].step()

        self.writer.add_scalar(f'cls/loss_on_synthetic', loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'cls/acc_on_synthetic', acc.item(), self.num_iters_done)

    def compute_centroid_loss(self, x_fake, y):
        groups = {}
        for x, l in zip(x_fake, y):
            if not l in groups: groups[l] = []
            groups[l].append(x)

        centroids_fake = {l: torch.stack(groups[l]).mean(dim=0) for l in groups}
        class_centroids = torch.tensor(self.class_centroids).to(x_fake.device).float()
        distances = [(c - class_centroids[l]).norm() for l, c in centroids_fake.items()]
        loss = sum(distances) # TODO: check that this thing is differentiable

        return loss / self.config.lll_setup.num_classes
