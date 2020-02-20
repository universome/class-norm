import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from firelab.base_trainer import BaseTrainer

from src.models.classifier import FeatClassifier
from src.models.feat_gan import FeatDiscriminator, ConditionalFeatDiscriminator, FeatGenerator
from src.utils.training_utils import validate_clf, construct_optimizer
from src.utils.losses import compute_gradient_penalty
from src.dataloaders.load_data import load_data


class GANTrainer(BaseTrainer):
    def init_models(self):
        if self.config.hp.model.gan_type == 'cgan':
            self.discriminator = ConditionalFeatDiscriminator(self.config.hp.model).to(self.device_name)
        elif self.config.hp.model.gan_type == 'acgan':
            self.discriminator = FeatDiscriminator(self.config.hp.model).to(self.device_name)
        else:
            raise NotImplementedError

        self.generator = FeatGenerator(self.config.hp.model).to(self.device_name)

    def init_dataloaders(self):
        self.ds_train, self.ds_test, _ = load_data(self.config.data)

        self.train_dataloader = DataLoader(self.ds_train, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)), shuffle=True)
        self.val_dataloader = DataLoader(self.ds_test, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)), shuffle=False)

        self.compute_prototypes()

    def compute_prototypes(self):
        compute_prototype = lambda ds, label: np.mean([x for x, y in ds if y == label], axis=0).tolist()

        self.prototypes = [compute_prototype(self.ds_train, y) for y in range(self.config.data.num_classes)]
        self.prototypes = torch.tensor(self.prototypes).to(self.device_name)

    def init_optimizers(self):
        self.optim = {
            'generator': construct_optimizer(self.generator.parameters(), self.config.hp.gen_optim),
            'discriminator': construct_optimizer(self.discriminator.parameters(), self.config.hp.discr_optim),
        }

    def train_on_batch(self, batch):
        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        self.discriminator_step(x, y)

        if self.num_iters_done % self.config.hp.num_discr_steps_per_gen_step == 0:
            self.generator_step()

    def discriminator_step(self, x: Tensor, y: Tensor):
        with torch.no_grad():
            x_fake = self.generator.sample(y) # TODO: try picking y randomly

        if self.config.hp.model.gan_type == 'cgan':
            adv_logits_on_real = self.discriminator(x, y)
            adv_logits_on_fake = self.discriminator(x_fake, y)
            grad_penalty = compute_gradient_penalty(self.discriminator, x, x_fake, y)
        elif self.config.hp.model.gan_type == 'acgan':
            adv_logits_on_real, cls_logits_on_real = self.discriminator(x)
            adv_logits_on_fake = self.discriminator.run_adv_head(x_fake)
            grad_penalty = compute_gradient_penalty(self.discriminator.run_adv_head, x, x_fake)
        else:
            raise NotImplementedError

        adv_loss = -adv_logits_on_real.mean() + adv_logits_on_fake.mean()
        total_loss = adv_loss + self.config.hp.loss_coefs.gp * grad_penalty

        if self.config.hp.model.gan_type == 'acgan':
            cls_loss = F.cross_entropy(cls_logits_on_real, y)
            cls_acc = (cls_logits_on_real.argmax(dim=1) == y).float().mean().detach().cpu()
            total_loss += self.config.hp.loss_coefs.discr_cls * cls_loss

            self.writer.add_scalar('discr/cls_loss', cls_loss.item(), self.num_iters_done)
            self.writer.add_scalar('discr/cls_acc', cls_acc.item(), self.num_iters_done)

        self.perform_optim_step(total_loss, 'discriminator')

        self.writer.add_scalar('discr/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_real', adv_logits_on_real.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_fake', adv_logits_on_fake.mean().item(), self.num_iters_done)

    def generator_step(self):
        num_samples_per_class = self.config.hp.gen_batch_size // self.config.data.num_classes
        y = torch.arange(self.config.data.num_classes).repeat(num_samples_per_class)
        y = y.to(self.device_name)
        x_fake = self.generator.sample(y)

        if self.config.hp.model.gan_type == 'acgan':
            adv_logits_on_fake, cls_logits_on_fake = self.discriminator(x_fake)

            adv_loss = -adv_logits_on_fake.mean()
            cls_loss = F.cross_entropy(cls_logits_on_fake, y)
            cls_acc = (cls_logits_on_fake.argmax(dim=1) == y).float().mean().detach().cpu()
            total_loss = adv_loss + self.config.hp.loss_coefs.gen_cls * cls_loss

            self.writer.add_scalar(f'gen/cls_loss', cls_loss.item(), self.num_iters_done)
            self.writer.add_scalar(f'gen/cls_acc', cls_acc.item(), self.num_iters_done)
        elif self.config.hp.model.gan_type == 'cgan':
            adv_logits_on_fake = self.discriminator(x_fake, y)

            adv_loss = -adv_logits_on_fake.mean()
            total_loss = adv_loss
        else:
            raise NotImplementedError

        if self.config.hp.loss_coefs.proto > 0:
            x_fake_mean = x_fake.view(self.config.data.num_classes, -1, *x_fake.shape[1:]).mean(dim=1)
            proto_loss = F.mse_loss(x_fake_mean, self.prototypes)
            total_loss += self.config.hp.loss_coefs.proto * proto_loss
            self.writer.add_scalar(f'gen/proto_loss', proto_loss.item(), self.num_iters_done)

        self.perform_optim_step(total_loss, 'generator')

        self.writer.add_scalar(f'gen/adv_loss', adv_loss.item(), self.num_iters_done)

    def after_training_hook(self):
        self.validate()

    def on_epoch_done(self):
        if self.num_epochs_done % 50 == 0:
            self.validate(val_freq_steps=self.config.hp.clf.max_num_steps)

    def validate(self, val_freq_steps: int=10):
        clf = FeatClassifier(self.config.hp.model).to(self.device_name)
        optim = torch.optim.Adam(clf.parameters(), **self.config.hp.cls_optim.kwargs)

        for step in range(1, self.config.hp.clf.max_num_steps + 1):
            with torch.no_grad():
                y = random.choices(range(self.config.data.num_classes), k=self.config.hp.clf.batch_size)
                y = torch.tensor(y).to(self.device_name)
                x = self.generator.sample(y)

            logits = clf(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean().cpu().detach()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % val_freq_steps == 0:
                val_loss, val_acc = validate_clf(clf, self.val_dataloader, self.device_name)

                print(f'[Step #{step:04d}] Train loss: {loss.item(): 0.4f}. Train acc: {acc.item(): 0.4f}')
                print(f'[Step #{step:04d}] Val loss  : {val_loss.item(): 0.4f}. Val acc  : {val_acc.item(): 0.4f}')

        print('Train loss/acc:', validate_clf(clf, self.train_dataloader, self.device_name))

    def perform_optim_step(self, loss, module_name: str, retain_graph: bool=False):
        self.optim[module_name].zero_grad()
        loss.backward(retain_graph=retain_graph)

        if self.config.hp.grad_clipping.has(module_name):
            grad_clip_val = self.config.hp.grad_clipping.has(module_name)
            grad_norm = nn.utils.clip_grad_norm_(getattr(self, module_name).parameters(), grad_clip_val)
            self.writer.add_scalar(f'grad_norms/{module_name}', grad_norm, self.num_iters_done)

        self.optim[module_name].step()