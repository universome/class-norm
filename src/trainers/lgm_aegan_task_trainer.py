import os
import random
from typing import Tuple, Iterable
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from firelab.config import Config

from src.utils.losses import compute_gradient_penalty
from src.utils.training_utils import prune_logits
from src.utils.model_utils import get_number_of_parameters
from src.models.lgm import LGM
from src.trainers.lgm_task_trainer import LGMTaskTrainer
from src.utils.weights_importance import compute_mse_grad, compute_diagonal_fisher
from src.utils.data_utils import construct_output_mask


class LGMAEGANTaskTrainer(LGMTaskTrainer):
    def _after_init_hook(self):
        super(LGMAEGANTaskTrainer, self)._after_init_hook()

        if self.task_idx > 0:
            self.statistics = self.get_previous_trainer().statistics
            self.embeddings = self.get_previous_trainer().embeddings
        else:
            self.statistics = {}
            self.embeddings = {}

    def construct_optimizer(self):
        return {
            'generator': self.construct_optimizer_from_config(self.get_parameters('generator'), self.config.hp.gen_optim),
            'discriminator': self.construct_optimizer_from_config(self.get_parameters('discriminator'), self.config.hp.discr_optim),
            'classifier': self.construct_optimizer_from_config(self.get_parameters('classifier'), self.config.hp.cls_optim),
        }

    def get_parameters(self, name: str) -> Iterable[nn.Parameter]:
        params_dict = {
            'generator': self.model.generator.parameters(),
            'discriminator': self.model.discriminator.get_adv_parameters(),
            'classifier': self.model.discriminator.get_cls_parameters(),
        }

        return params_dict[name]

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        if self.num_epochs_done < self.config.hp.num_cls_epochs:
            self.classifier_step(x, y)
        else:
            self.generator_step(y)
            self.discriminator_step(x, y)

    def build_memory_for_the_current_dataset(self):
        self.model.eval()

        with torch.no_grad():
            class_to_embs = {c: torch.empty(0, feat_size) for c in self.classes}

            for x, y in self.train_dataloader:
                embs = self.model.discriminator.adv_body(x.to(self.device_name)).cpu()
                all_embs = torch.cat([all_embs, embs])

    def update_memory_for_seen_datasets(self):
        self.model.eval()

        with torch.no_grad():
            for y in range(self.learned_classes):
                # Recomputing embeddings
                feats = self.model.generator(self.embeddings[y])
                new_embs = self.model.discriminator.adv_body(feats)
                new_embs = new_embs / new_embs.norm(dim=1)

                n = len(new_embs)
                mean = new_embs.mean(dim=0)
                cov = (new_embs - mean.unsqueeze(1)).t_() @ (new_embs - mean.unsqueeze(1)) / (n - 1)

                # Updating statistics
                self.statistics[y]['mean'] = mean
                self.statistics[y]['cov'] = cov

                # Updating embeddings
                dist = MultivariateNormal(mean, covariance_matrix=self.statistics[y]['cov'])
                self.embeddings[y] = dist.sample(mean.size(0), self.config.hp.num_embeddings_per_class)
                self.embeddings[y] = self.embeddings[y] / self.embeddings[y].norm(dim=1)
                self.embeddings[y] = self.embeddings[y].cpu()

    def on_epoch_done(self):
        if self.num_epochs_done == self.config.hp.num_cls_epochs:
            self.collect_statistics()
            self.collect_embeddings()

    def discriminator_step(self, imgs: Tensor, y: Tensor):
        with torch.no_grad():
            x = self.model.embedder(imgs)
            z = self.model.generator.sample_noise(imgs.size(0)).to(self.device_name)
            x_fake = self.model.generator(z, y) # TODO: try picking y randomly

        adv_logits_on_real = self.model.discriminator.run_adv_head(x)
        adv_logits_on_fake = self.model.discriminator.run_adv_head(x_fake)

        adv_loss = -adv_logits_on_real.mean() + adv_logits_on_fake.mean()
        grad_penalty = compute_gradient_penalty(self.model.discriminator.run_adv_head, x, x_fake)

        total_loss = adv_loss + self.config.hp.loss_coefs.gp * grad_penalty

        self.perform_optim_step(total_loss, 'discriminator')

        self.writer.add_scalar('discr/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_real', adv_logits_on_real.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_fake', adv_logits_on_fake.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/grad_penalty', grad_penalty.item(), self.num_iters_done)

    def generator_step(self, y: Tensor):
        if self.num_iters_done % self.config.hp.num_discr_steps_per_gen_step != 0: return

        z = self.model.generator.sample_noise(y.size(0)).to(self.device_name)
        x_fake = self.model.generator(z, y)
        adv_logits_on_fake, cls_logits_on_fake = self.model.discriminator(x_fake)
        pruned_logits_on_fake = prune_logits(cls_logits_on_fake, self.output_mask)

        adv_loss = -adv_logits_on_fake.mean()
        distillation_loss = self.knowledge_distillation_loss()
        cls_loss = F.cross_entropy(pruned_logits_on_fake, y)
        cls_acc = (pruned_logits_on_fake.argmax(dim=1) == y).float().mean().detach().cpu()

        total_loss = adv_loss \
            + self.config.hp.loss_coefs.distill * distillation_loss \
            + self.config.hp.loss_coefs.gen_cls * cls_loss

        self.perform_optim_step(total_loss, 'generator')

        self.writer.add_scalar(f'gen/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/distillation_loss', distillation_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/cls_acc', cls_acc.item(), self.num_iters_done)

    def classifier_step(self, x: Tensor, y: Tensor):
        pruned = self.model.compute_pruned_predictions(x, self.output_mask)
        curr_loss = self.criterion(pruned, y)
        curr_acc = (pruned.argmax(dim=1) == y).float().mean().detach().cpu()

        if len(self.statistics) > 0:
            rehearsal_loss, rehearsal_acc = self.compute_rehearsal_loss()
            total_loss = curr_loss + self.config.hp.rehearsal.loss_coef * rehearsal_loss

            self.writer.add_scalar('cls/rehearsal_loss', rehearsal_loss.item(), self.num_iters_done)
            self.writer.add_scalar('cls/rehearsal_acc', rehearsal_acc.item(), self.num_iters_done)
        else:
            total_loss = curr_loss

        self.perform_optim_step(total_loss, 'classifier')

        self.writer.add_scalar('cls/curr_oss', curr_loss.item(), self.num_iters_done)
        self.writer.add_scalar('cls/curr_acc', curr_acc.item(), self.num_iters_done)

    def compute_rehearsal_loss(self):
        y = random.sample(self.learned_classes, k=self.config.hp.rehearsal.batch_size)
        embeddings = [random.choice(self.statistics[c]) for c in y]

        y = torch.tensor(y).to(self.device_name)
        embeddings = torch.stack(embeddings).to(self.device_name)

        feats = self.model.generator(embeddings)
        logits = self.model.compute_pruned_predictions(feats, self.seen_classes_mask)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean().detach().cpu()

        return loss, acc
