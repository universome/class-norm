import os
from copy import deepcopy
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.losses import compute_kld_with_standard_gaussian, compute_kld_between_diagonal_gaussians
from src.trainers.task_trainer import TaskTrainer
from src.models.vae import FeatVAEClassifier
from src.dataloaders.utils import extract_features


class GenMemTaskTrainer(TaskTrainer):
    """
    Training generative memory until it overfits completely on each task
    """
    def _after_init_hook(self):
        self.mse_criterion = nn.MSELoss()
        self.prev_model = deepcopy(self.model).to(self.device_name).eval()

        if self.config.hp.get('reinit_at_each_task'):
            self.model = FeatVAEClassifier(self.config.hp.model_config, self.main_trainer.class_attributes).to(self.device_name)
            self.main_trainer.model = self.model

        self.current_classes = self.main_trainer.class_splits[self.task_idx]
        self.previously_seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx]).tolist()
        self.seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx + 1]).tolist()
        self.seen_classes_mask = np.array([c in self.seen_classes for c in range(self.config.data.num_classes)])
        self.writer = SummaryWriter(os.path.join(self.main_trainer.paths.logs_path, f'task_{self.task_idx}'))
        self.optim = self.construct_optimizer()

    def _before_train_hook(self):
        if self.config.hp.model_config.has('feat_extractor'):
            self.train_feat_extractor()
            self.extract_features()

    def _after_train_hook(self):
        self.train_classifier()

    def construct_optimizer(self) -> Dict[str, nn.Module]:
        optims = {
            'vae': torch.optim.Adam(self.model.vae.parameters(), **self.config.hp.optim_kwargs.to_dict()),
            'classifier': torch.optim.Adam(self.model.classifier.parameters(), **self.config.hp.optim_kwargs.to_dict()),
        }

        if self.config.hp.model_config.has('feat_extractor'):
            optims['feat_extractor'] = torch.optim.Adam(
                self.model.feat_extractor.parameters(),
                **self.config.hp.optim_kwargs.to_dict())

        return optims

    def train_on_batch(self, batch):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        self.train_vae_on_batch(x, y)

    def train_vae_on_batch(self, x, y):
        x_rec, mean, log_var = self.model.vae(x, y)
        rec_loss = self.mse_criterion(x_rec, x)
        #kld = compute_kld_with_standard_gaussian(mean, log_var)
        prior_mean, prior_log_var = self.model.vae.get_prior_distribution(y)
        kld = compute_kld_between_diagonal_gaussians(mean, log_var, prior_mean, prior_log_var)

        total_loss = rec_loss + self.config.hp.kl_term_coef * kld

        if self.task_idx > 0:
            enc_distill_loss, dec_distill_loss = self.knowledge_distillation_loss()
            total_loss += self.config.hp.enc_distill_loss_coef * enc_distill_loss \
                        + self.config.hp.dec_distill_loss_coef * dec_distill_loss

            self.writer.add_scalar('vae/enc_distill_loss', enc_distill_loss.item(), self.num_iters_done)
            self.writer.add_scalar('vae/dec_distill_loss', dec_distill_loss.item(), self.num_iters_done)

        self.optim['vae'].zero_grad()
        total_loss.backward()
        self.optim['vae'].step()

        self.writer.add_scalar('vae/rec_loss', rec_loss.item(), self.num_iters_done)
        self.writer.add_scalar('vae/kl_loss', kld.item(), self.num_iters_done)

    def knowledge_distillation_loss(self) -> Tuple[Tensor, Tensor]:
        y = np.random.choice(self.previously_seen_classes, self.config.hp.distillation_batch_size)
        y = torch.tensor(y).to(self.device_name).long()

        with torch.no_grad():
            x = self.prev_model.vae.generate(y)
            mean_old, log_var_old = self.prev_model.vae.encode(x, y)
            z = self.prev_model.vae.sample(mean_old, log_var_old)
            x_rec_old = self.prev_model.vae.decode(z, y)

        mean_new, log_var_new = self.model.vae.encode(x, y)
        x_rec_new = self.model.vae.decode(z, y)

        enc_distill_loss = self.mse_criterion(
            torch.cat([mean_new, log_var_new], dim=1),
            torch.cat([mean_old, log_var_old], dim=1))
        dec_distill_loss = self.mse_criterion(x_rec_new, x_rec_old)

        return enc_distill_loss, dec_distill_loss

    def train_classifier(self):
        for clf_train_iter in tqdm(range(self.config.hp.clf_training.num_iters), desc=f'Task {self.task_idx} [clf]'):
            with torch.no_grad():
                y = np.random.choice(self.seen_classes, self.config.hp.clf_training.batch_size)
                y = torch.tensor(y).to(self.device_name)
                x = self.model.vae.generate(y)

            pruned_logits = self.model.compute_pruned_predictions(x, self.seen_classes_mask)
            loss = self.criterion(pruned_logits, y)
            acc = (pruned_logits.argmax(dim=1) == y).float().mean()

            self.optim['classifier'].zero_grad()
            loss.backward()
            self.optim['classifier'].step()

            self.writer.add_scalar('clf/loss', loss.item(), clf_train_iter)
            self.writer.add_scalar('clf/acc', acc.item(), clf_train_iter)

    def train_feat_extractor(self):
        self.model.train()
        self.num_feat_ext_iters_done = 0

        for epoch in range(1, self.config.hp.feat_extractor.num_epochs + 1):
            for batch in tqdm(self.train_dataloader, desc=f'[task #{self.task_idx}/feat_ext epoch #{epoch}]'):
                x = torch.tensor(batch[0]).to(self.device_name)
                y = torch.tensor(batch[1]).to(self.device_name)
                self.train_feat_extractor_on_batch(x, y)
                self.num_feat_ext_iters_done += 1

    def train_feat_extractor_on_batch(self, x, y):
        pruned_logits = self.model.feat_extractor.compute_pruned_predictions(x, self.output_mask)
        loss = self.criterion(pruned_logits, y)
        acc = (pruned_logits.argmax(dim=1) == y).float().mean()

        self.optim['feat_extractor'].zero_grad()
        loss.backward()
        self.optim['feat_extractor'].step()

        self.writer.add_scalar('feat_ext/loss', loss.item(), self.num_feat_ext_iters_done)
        self.writer.add_scalar('feat_ext/acc', acc.item(), self.num_feat_ext_iters_done)

    def extract_features(self):
        self.model.eval()

        imgs_train = extract_features([x for x, _ in self.task_ds_train], self.model.feat_extractor.embedder)
        imgs_test = extract_features([x for x, _ in self.task_ds_test], self.model.feat_extractor.embedder)

        self.task_ds_train = [(x, y) for x, (_, y) in zip(imgs_train, self.task_ds_train)]
        self.task_ds_test = [(x, y) for x, (_, y) in zip(imgs_test, self.task_ds_test)]
        self.init_dataloaders()
