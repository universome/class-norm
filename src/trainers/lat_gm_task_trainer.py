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
from firelab.config import Config

from src.utils.losses import compute_gradient_penalty
from src.utils.lll import prune_logits
from src.utils.model_utils import get_number_of_parameters
from src.models.lat_gm import LatGM
from src.trainers.task_trainer import TaskTrainer
from src.utils.weights_importance import compute_mse_grad, compute_diagonal_fisher


class LatGMTaskTrainer(TaskTrainer):
    BaseModel = LatGM

    def _after_init_hook(self):
        prev_trainer = self.get_previous_trainer()

        self.prev_model = self.BaseModel(self.config.hp.model, self.model.attrs).to(self.device_name)
        self.prev_model.load_state_dict(deepcopy(self.model.state_dict()))
        self.learned_classes = np.unique(self.main_trainer.class_splits[:self.task_idx]).tolist()
        self.seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx + 1]).tolist()
        self.writer = SummaryWriter(os.path.join(self.main_trainer.paths.logs_path, f'task_{self.task_idx}'), flush_secs=5)

        if not prev_trainer is None:
            self.weights_prev = torch.cat([p.data.view(-1) for p in self.model.embedder.parameters()])

            if self.config.hp.reg_strategy == 'mas':
                self.mse_grad = compute_mse_grad(self.model, self.train_dataloader, prev_trainer.output_mask)
            elif self.config.hp.reg_strategy == 'ewc':
                self.fisher = compute_diagonal_fisher(self.model, self.train_dataloader, prev_trainer.output_mask)
            else:
                raise NotImplementedError(f'Unknown regularization strategy: {self.config.hp.reg_strategy}')

    def construct_optimizer(self):
        return {
            'generator': self.construct_optimizer_from_config(self.model.generator.parameters(), self.config.hp.gen_optim),
            'discriminator': self.construct_optimizer_from_config(self.model.discriminator.parameters(), self.config.hp.discr_optim),
            'classifier': self.construct_optimizer_from_config(self.model.embedder.parameters(), self.config.hp.clf_optim),
        }

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

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray]):
        self.model.train()

        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        if self.num_iters_done % self.config.hp.num_discr_steps_per_gen_step == 0:
            self.generator_step(y)

        if self.config.hp.get('creativity.enabled') and self.num_iters_done > self.config.hp.creativity.start_iter:
            self.creativity_step()

        self.discriminator_step(x, y)
        self.classifier_step(x, y)

    def discriminator_step(self, imgs: Tensor, y: Tensor):
        with torch.no_grad():
            x = self.model.embedder(imgs)
            z = self.model.generator.sample_noise(imgs.size(0)).to(self.device_name)
            x_fake = self.model.generator(z, y) # TODO: try picking y randomly

        adv_logits_on_real = self.model.discriminator.run_adv_head(x)
        adv_logits_on_fake = self.model.discriminator.run_adv_head(x_fake)

        adv_loss = -adv_logits_on_real.mean() + adv_logits_on_fake.mean()
        grad_penalty = compute_gradient_penalty(self.model.discriminator.run_adv_head, x, x_fake)
        discr_loss_total = adv_loss + self.config.hp.gp_coef * grad_penalty

        self.optim['discriminator'].zero_grad()
        discr_loss_total.backward()
        self.optim['discriminator'].step()

        self.writer.add_scalar('discr/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_real', adv_logits_on_real.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/mean_pred_fake', adv_logits_on_fake.mean().item(), self.num_iters_done)
        self.writer.add_scalar('discr/grad_penalty', grad_penalty.item(), self.num_iters_done)

    def generator_step(self, y: Tensor):
        z = self.model.generator.sample_noise(y.size(0)).to(self.device_name)
        x_fake = self.model.generator(z, y)
        adv_logits_on_fake, cls_logits_on_fake = self.model.discriminator(x_fake)
        pruned_logits_on_fake = prune_logits(cls_logits_on_fake, self.output_mask)

        adv_loss = -adv_logits_on_fake.mean()
        distillation_loss = self.knowledge_distillation_loss()
        cls_loss = F.cross_entropy(pruned_logits_on_fake, y)
        cls_acc = (pruned_logits_on_fake.argmax(dim=1) == y).float().mean().detach().cpu()

        total_loss = adv_loss \
            + self.config.hp.distill_loss_coef * distillation_loss \
            + self.config.hp.cls_loss_coef_gen * cls_loss

        self.optim['generator'].zero_grad()
        total_loss.backward()
        self.optim['generator'].step()

        self.writer.add_scalar(f'gen/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/distillation_loss', distillation_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar(f'gen/cls_acc', cls_acc.item(), self.num_iters_done)

    def knowledge_distillation_loss(self) -> Tensor:
        if len(self.learned_classes) == 0: return torch.tensor(0.)

        num_samples_per_class = self.config.hp.distill_batch_size // len(self.learned_classes)
        assert num_samples_per_class >= 0, "Distillation batch size is too small to capture all the classes."
        y = np.tile(np.array(self.learned_classes), num_samples_per_class)
        y = torch.tensor(y).to(self.device_name)
        z = self.model.generator.sample_noise(len(y)).to(self.device_name)
        outputs_teacher = self.prev_model.generator(z, y).view(len(y), -1)
        outputs_student = self.model.generator(z, y).view(len(y), -1)
        loss = torch.norm(outputs_teacher - outputs_student, dim=1).pow(2).mean()

        return loss / len(self.learned_classes)

    def sample_train_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = min(len(self.task_ds_train), self.config.hp.batch_size)
        idx = random.sample(range(len(self.task_ds_train)), batch_size)
        x, y = zip(*[self.task_ds_train[i] for i in idx])

        return x, y

    def classifier_step(self, x: Tensor, y: Tensor):
        pruned = self.model.compute_pruned_predictions(x, self.output_mask)
        loss = self.criterion(pruned, y)
        acc = (pruned.argmax(dim=1) == y).float().mean().detach().cpu()

        if self.task_idx > 0:
            reg = self.compute_classifier_reg()
            loss += self.config.hp.synaptic_strength * reg

        self.optim['classifier'].zero_grad()
        loss.backward()
        self.optim['classifier'].step()

        self.writer.add_scalar('clf/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar('clf/acc', acc.item(), self.num_iters_done)

    def compute_classifier_reg(self) -> Tensor:
        if self.config.hp.reg_strategy == 'mas':
            return self.compute_classifier_mas_reg()
        elif self.config.hp.reg_strategy == 'ewc':
            return self.compute_classifier_ewc_reg()
        else:
            raise NotImplementedError(f'Unknown regularization strategy: {self.config.hp.reg_strategy}')

    def compute_classifier_mas_reg(self) -> Tensor:
        weights_curr = torch.cat([p.view(-1) for p in self.model.embedder.parameters()])
        reg = torch.dot((weights_curr - self.weights_prev).pow(2), self.mse_grad)

        return reg

    def compute_classifier_ewc_reg(self) -> Tensor:
        embedder_size = get_number_of_parameters(self.model.embedder)
        keep_prob = self.config.hp.get('fisher_keep_prob', 1.)
        weights_curr = torch.cat([p.view(-1) for p in self.model.embedder.parameters()])

        body_fisher = self.fisher[:embedder_size]
        body_weights_curr = weights_curr[:embedder_size]
        body_weights_prev = self.weights_prev[:embedder_size]

        if keep_prob < 1:
            body_fisher = F.dropout(body_fisher, keep_prob)

        reg = torch.dot((body_weights_curr - body_weights_prev).pow(2), body_fisher)

        return reg

    def creativity_step(self) -> Tensor:
        # Computes creativity loss. Creativity loss is about making Classifier unsure about samples,
        # generated from unseen attributes, but these samples should seem real to Discriminator.
        n_samples = self.config.hp.creativity.hall_batch_size

        # Sample twice as more examples
        y_from = np.random.choice(self.seen_classes, size=n_samples * 3, replace=True)
        y_to = np.random.choice(self.seen_classes, size=n_samples * 3, replace=True)

        # Let's remove overlapping classes for cleaner hallucinations
        overlaps = y_from == y_to
        y_from, y_to = y_from[~overlaps], y_to[~overlaps]

        # Removing extra samples if they exist
        y_from, y_to = y_from[:n_samples], y_to[:n_samples]
        assert len(y_from) == len(y_to)

        # Interpolating attributes
        alpha = np.random.rand(len(y_from)).astype(np.float32)
        alpha = 0.6 * alpha + 0.2 # Now it's random uniform on [0.2, 0.8]
        alpha = torch.tensor(alpha).to(self.device_name).unsqueeze(1)
        hall_attrs = self.model.attrs[y_from] * alpha + self.model.attrs[y_to] * (1 - alpha)
        hall_attrs = torch.tensor(hall_attrs).to(self.device_name)

        z = self.model.generator.sample_noise(len(hall_attrs)).to(self.device_name)
        hall_samples = self.model.generator.forward_with_attr(z, hall_attrs)
        adv_logits, cls_logits = self.model.discriminator(hall_samples)
        cls_log_probs = F.log_softmax(cls_logits, dim=1)

        adv_loss = -adv_logits.mean()
        entropy = (cls_log_probs.exp() @ cls_log_probs.t()).sum(dim=1).mean()

        loss = self.config.hp.creativity.adv_coef * adv_loss + self.config.hp.creativity.entropy_coef * entropy

        self.optim['generator'].zero_grad()
        loss.backward()
        self.optim['generator'].step()

        self.writer.add_scalar('creativity/adv_loss', adv_loss.item(), self.num_iters_done)
        self.writer.add_scalar('creativity/entropy', entropy.item(), self.num_iters_done)
