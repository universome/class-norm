import os
import random
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
from src.models.lat_gm import LatGM
from src.trainers.task_trainer import TaskTrainer
from src.utils.weights_importance import compute_mse_grad, compute_diagonal_fisher


class LatGMTaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        prev_trainer = self.get_previous_trainer()

        self.prev_model = LatGM(self.config.hp.model_config, self.model.attrs).to(self.device_name)
        self.prev_model.load_state_dict(deepcopy(self.model.state_dict()))
        self.seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx]).tolist()
        self.writer = SummaryWriter(os.path.join(self.main_trainer.paths.logs_path, f'task_{self.task_idx}'), flush_secs=5)

        if not prev_trainer is None:
            self.weights_prev = torch.cat([p.data.view(-1) for p in self.model.classifier.parameters()])

            if self.config.hp.reg_strategy == 'mas':
                self.mse_grad = compute_mse_grad(self.model.classifier, self.train_dataloader, prev_trainer.output_mask)
            elif self.config.hp.reg_strategy == 'ewc':
                self.fisher = compute_diagonal_fisher(self.model.classifier, self.train_dataloader, prev_trainer.output_mask)
            else:
                raise NotImplementedError(f'Unknown regularization strategy: {self.config.hp.reg_strategy}')


    def construct_optimizer(self):
        return {
            'generator': torch.optim.Adam(self.model.generator.parameters(), **self.config.hp.gen_optim.kwargs.to_dict()),
            'discriminator': torch.optim.Adam(self.model.discriminator.parameters(), **self.config.hp.discr_optim.kwargs.to_dict()),
            'classifier': torch.optim.Adam(self.model.classifier.parameters(), **self.config.hp.discr_optim.kwargs.to_dict())
        }

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        if self.num_iters_done % self.config.hp.num_discr_steps_per_gen_step == 0:
            self.generator_step(y)

        self.discriminator_step(x, y)

    def discriminator_step(self, imgs: Tensor, y: Tensor):
        with torch.no_grad():
            z = self.model.generator.sample_noise(x.size(0)).to(self.device_name)
            x_fake = self.model.generator(z, self.model[y]) # TODO: try picking y randomly
            x = self.model.embed(imgs)

        adv_logits_on_real = self.model.discriminator(x)
        adv_logits_on_fake = self.model.discriminator(x_fake)

        adv_loss = -adv_logits_on_real.mean() + adv_logits_on_fake.mean()
        grad_penalty = compute_gradient_penalty(self.model.discriminator, x, x_fake)
        discr_loss_total = adv_loss + self.config.hp.gp_coef * grad_penalty

        self.optim['discriminator'].zero_grad()
        discr_loss_total.backward()
        self.optim['discriminator'].step()

        self.writer.add_scalar('discr/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_real', logits_on_real.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_fake', logits_on_fake.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/grad_penalty', grad_penalty.item(), self.num_iters_done)

    def generator_step(self, y: Tensor):
        z = self.model.generator.sample_noise(y.size(0)).to(self.device_name)
        x_fake = self.model.generator(z, y)
        discr_logits_on_fake, cls_logits_on_fake = self.model.discriminator(x_fake)

        adv_loss = -discr_logits_on_fake.mean()
        cls_loss = F.cross_entropy(prune_logits(cls_logits_on_fake, self.output_mask), y)
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
        if len(self.seen_classes) == 0: return torch.tensor(0.)

        num_samples_per_class = self.config.hp.distill_batch_size // len(self.seen_classes)
        y = np.array(self.previously_seen_classes).tile(num_samples_per_class)
        y = torch.tensor(y).to(self.device_name)
        z = self.model.generator.sample_noise(len(y)).to(self.device_name)
        outputs_teacher = self.prev_model.generator(z, y).view(len(y), -1)
        outputs_student = self.model.generator(z, y).view(len(y), -1)
        loss = torch.norm(outputs_teacher - outputs_student, dim=1).pow(2).mean()

        return loss / len(self.seen_classes)

    def start(self):
        self._before_train_hook()

        assert self.is_trainable, "We do not have enough conditions to train this Task Trainer" \
                                  "(for example, previous trainers was not finished or this trainer was already run)"

        for i in tqdm(range(self.config.hp.num_iters_per_task), desc=f'Task #{self.task_idx} iteration'):
            batch = self.sample_train_batch()
            self.train_on_batch(batch)
            self.num_iters_done += 1
            self.run_after_iter_done_callbacks()

        self._after_train_hook()

    def sample_train_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = min(len(self.task_ds_train), self.config.hp.batch_size)
        idx = random.sample(range(len(self.task_ds_train)), batch_size)
        x, y = zip(*[self.task_ds_train[i] for i in idx])

        return x, y

    def train_on_batch(self, batch:Tuple[Tensor, Tensor]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        pruned_logits = self.model.compute_pruned_predictions(x, self.output_mask)
        loss = self.criterion(pruned_logits, y)

        if self.task_idx > 0:
            reg = self.compute_classifier_reg()
            loss += self.config.hp.synaptic_strength * reg

        self.optim['classifier'].zero_grad()
        loss.backward()
        self.optim['classifier'].step()

    def compute_classifier_reg(self) -> Tensor:
        if self.config.hp.reg_strategy == 'mas':
            return self.compute_classifier_mas_reg()
        elif self.config.hp.reg_strategy == 'ewc':
            return self.compute_classifier_ewc_reg()
        else:
            raise NotImplementedError(f'Unknown regularization strategy: {self.config.hp.reg_strategy}')

    def compute_classifier_mas_reg(self) -> Tensor:
        weights_curr = torch.cat([p.view(-1) for p in self.model.classifier.embedder.parameters()])
        reg = torch.dot((weights_curr - self.weights_prev).pow(2), self.mse_grad)

        return reg

    def compute_classifier_ewc_reg(self) -> Tensor:
        head_size = self.model.classifier.get_head_size()
        keep_prob = self.config.hp.get('fisher_keep_prob', 1.)
        weights_curr = torch.cat([p.view(-1) for p in self.model.classifier.parameters()])

        body_fisher = self.fisher[:-head_size]
        body_weights_curr = weights_curr[:-head_size]
        body_weights_prev = self.weights_prev[:-head_size]

        if keep_prob < 1:
            body_fisher = F.dropout(body_fisher, keep_prob)

        reg = torch.dot((body_weights_curr - body_weights_prev).pow(2), body_fisher)

        return reg

    def creativity_loss(self) -> Tensor:
        # Creativity loss is about making Classifier unsure about samples, generated from unseen attributes
        # But these samples should seem real to Discriminator
        alpha = torch.rand(self.config.hp.creativity.hall_batch_size).to(self.device_name)
        alpha = 0.6 * alpha + 0.2 # Now it's random uniform on [0.2, 0.8]
        y = random.choices(self.seen_classes)
