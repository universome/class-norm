import os
import random
from copy import deepcopy
from typing import Tuple, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.losses import compute_kld_with_standard_gaussian, compute_kld_between_diagonal_gaussians
from src.trainers.task_trainer import TaskTrainer
from src.trainers.lat_gm_task_trainer import LatGMTaskTrainer
from src.models.lat_gm_vae import LatGMVAE
from src.dataloaders.utils import extract_features


class LatGMVAETaskTrainer(LatGMTaskTrainer):
    """
    Training generative memory until it overfits completely on each task
    """
    BaseModel = LatGMVAE

    def construct_optimizer(self) -> Dict:
        return {
            'vae': torch.optim.Adam(self.get_parameters('vae'), **self.config.hp.vae_optim.kwargs.to_dict()),
            'embedder': torch.optim.Adam(self.get_parameters('embedder'), **self.config.hp.embedder_optim.kwargs.to_dict()),
            'classifier': torch.optim.Adam(self.get_parameters('classifier'), **self.config.hp.clf_optim.kwargs.to_dict()),
        }

    def get_parameters(self, name: str) -> Iterable[nn.Parameter]:
        params_dict = {
            'vae': self.model.vae.parameters(),
            'classifier': self.model.classifier.parameters(),
            'embedder': self.model.embedder.parameters(),
        }

        return params_dict[name]

    def train_on_batch(self, batch):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        if self.config.hp.num_vae_epochs < self.num_epochs_done:
            self.vae_step(x, y)
        else:
            self.classifier_step(x, y)

    def vae_step(self, imgs, y):
        with torch.no_grad(): x = self.model.embedder(imgs)

        x_rec, mean, log_var = self.model.vae(x, y)
        rec_loss = F.mse_loss(x_rec, x)
        #kld = compute_kld_with_standard_gaussian(mean, log_var)
        prior_mean, prior_log_var = self.model.vae.get_prior_distribution(y)
        kld = compute_kld_between_diagonal_gaussians(mean, log_var, prior_mean, prior_log_var)

        total_loss = rec_loss + self.config.hp.kl_term_coef * kld

        if self.task_idx > 0:
            enc_distill_loss, dec_distill_loss = self.compute_distillation_loss()
            total_loss += self.config.hp.distillation.enc_loss_coef * enc_distill_loss \
                        + self.config.hp.distillation.dec_loss_coef * dec_distill_loss

            self.writer.add_scalar('vae/enc_distill_loss', enc_distill_loss.item(), self.num_iters_done)
            self.writer.add_scalar('vae/dec_distill_loss', dec_distill_loss.item(), self.num_iters_done)

        self.optim['vae'].zero_grad()
        total_loss.backward()
        self.optim['vae'].step()

        self.writer.add_scalar('vae/rec_loss', rec_loss.item(), self.num_iters_done)
        self.writer.add_scalar('vae/kl_loss', kld.item(), self.num_iters_done)

    def compute_distillation_loss(self) -> Tuple[Tensor, Tensor]:
        y = np.random.choice(self.learned_classes, self.config.hp.distillation.batch_size)
        y = torch.tensor(y).to(self.device_name).long()

        with torch.no_grad():
            x = self.prev_model.vae.generate(y)
            mean_old, log_var_old = self.prev_model.vae.encode(x, y)
            z = self.prev_model.vae.sample(mean_old, log_var_old)
            x_rec_old = self.prev_model.vae.decode(z, y)

        mean_new, log_var_new = self.model.vae.encode(x, y)
        x_rec_new = self.model.vae.decode(z, y)

        enc_distill_loss = F.mse_loss(
            torch.cat([mean_new, log_var_new], dim=1),
            torch.cat([mean_old, log_var_old], dim=1))
        dec_distill_loss = F.mse_loss(x_rec_new, x_rec_old)

        return enc_distill_loss, dec_distill_loss
