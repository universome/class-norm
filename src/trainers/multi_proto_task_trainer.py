import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch import autograd

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits, normalize
from src.utils.data_utils import sample_instances_for_em
from src.utils.losses import (
    compute_mean_distance,
    compute_gdpp_loss,
    compute_mmd_loss,
    compute_diagonal_cov_reg
)

class MultiProtoTaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        # if self.config.hp.get('reverse_clf.loss_coef'):
        #     self._init_core_images(self.config.hp.reverse_clf.n_imgs_per_class)
        # elif self.config.hp.get('pull_golden_protos.loss_coef'):
        #     self._init_core_images(self.config.hp.pull_golden_protos.n_imgs_per_class)

        if self.config.hp.get('rehearsal.loss_coef'):
            self.init_episodic_memory()

    def update_episodic_memory(self):
        if not self.config.hp.has('rehearsal'):
            return

        for c in self.classes:
            mem = sample_instances_for_em(self.task_ds_train, c, self.config.hp.rehearsal.n_samples_per_class)
            self.episodic_memory.extend([xy for xy in mem])

    # def _init_core_images(self, n_imgs_per_class: int):
    #     core_images = [[x for x, y in self.task_ds_train if y == c][:n_imgs_per_class] for c in self.classes]

    #     self.core_images = torch.tensor(core_images) # [n_task_classes, n_imgs_per_class, *image_shape]
    #     self.core_images = self.core_images.view(-1, *self.core_images.shape[2:])

    def train_on_batch(self, batch):
        self.model.train()

        loss = 0.
        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        # if self.model.head.config.get('dae.enabled'):
        #     with torch.no_grad():
        #         feats = self.model.embedder(x)
        #     logits = self.model.head(feats)
        #     feats_rec = self.model.head.compute_dae_reconstructions(feats, y)

        #     rec_loss = torch.norm(feats_rec - feats, dim=1).mean()
        #     loss += self.model.head.config.dae.loss_coef * rec_loss
        #     self.writer.add_scalar('rec_loss', rec_loss.item(), self.num_iters_done)

        # if self.config.hp.get('triplet_loss.enabled'):
        #     feats = self.model.embedder(x) # [batch_size, hid_dim]
        #     protos = self.model.head.generate_prototypes() # [n_protos, n_classes, hid_dim]

        #     feats = normalize(feats) # [batch_size, hid_dim]
        #     protos = normalize(protos)

        #     batch_size = feats.size(0)
        #     n_protos, _, hid_dim = protos.size()
        #     n_curr_classes = len(self.classes)

        #     protos = protos[:, self.classes, :] # [n_protos, n_curr_classes, hid_dim]
        #     protos = protos.view(n_protos * n_curr_classes, hid_dim).permute(1, 0) # [hid_dim, n_protos * n_curr_classes]
        #     distances = (feats.unsqueeze(2) - protos.unsqueeze(0)).norm(dim=1).pow(2) # [batch_size, n_protos * n_curr_classes]
        #     distances = distances.view(batch_size, n_protos, n_curr_classes)
        #     classes_in_batch = y.view(batch_size, 1, 1).repeat(1, n_protos, n_curr_classes)
        #     classes_generated = torch.tensor(self.classes).to(self.device_name).view(1, 1, n_curr_classes).repeat(batch_size, n_protos, 1)
        #     positive_distances = distances.masked_select(classes_in_batch == classes_generated)
        #     negative_distances = distances.masked_select(classes_in_batch != classes_generated)

        #     # Now we should remove too easy negative distances
        #     smallest_negative_distances_idx = negative_distances.sort()[1][:len(positive_distances)]
        #     negative_distances = negative_distances[smallest_negative_distances_idx]
        #     #triplet_loss = torch.max(positive_distances - negative_distances + self.config.hp.triplet_loss.margin, 0)
        #     # TODO: well, this is actually not a triplet loss...
        #     triplet_loss = positive_distances.mean() - negative_distances.mean()

        #     loss += triplet_loss * self.config.hp.triplet_loss.coef
        #     # print('triplet', triplet_loss.item())

        #     self.writer.add_scalar('triplet_loss/loss', triplet_loss.item(), self.num_iters_done)
        #     self.writer.add_scalar('triplet_loss/positive_mean_dist', positive_distances.mean().item(), self.num_iters_done)
        #     self.writer.add_scalar('triplet_loss/negative_mean_dist', negative_distances.mean().item(), self.num_iters_done)

        # if self.config.hp.get('protos_clf_loss_coef') or self.config.hp.get('push_protos_apart_loss_coef'):
        #     logits, protos = self.model(x, return_protos=True)
        # else:
        feats = self.model.embedder(x)
        logits = self.model.head(feats)

        if self.model.head.config.get('aggregation_type') == 'individual_losses':
            n_protos = logits.size(0) // y.size(0)
            batch_size = y.size(0)
            y = y.view(batch_size, 1).repeat(1, n_protos).view(batch_size * n_protos)
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)
        elif self.model.head.config.get('aggregation_type') == 'gmm':
            logits_for_true = logits[torch.arange(logits.size(0)), y] # [batch_size]
            logits_pruned = prune_logits(logits, self.output_mask) # [batch_size, n_classes]
            log_evidence = logits_pruned.logsumexp(dim=1) # [batch_size]
            cls_loss = -(logits_for_true - log_evidence).mean()
        else:
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)

        loss += cls_loss * self.config.get('cls_loss.coef', 1.0)

        if self.task_idx > 0 and self.config.hp.get('rehearsal.loss_coef'):
            rehearsal_loss = self.compute_rehearsal_loss()
            loss += self.config.hp.rehearsal.loss_coef * rehearsal_loss
            self.writer.add_scalar('rehearsal_loss', rehearsal_loss.item(), self.num_iters_done)

        # if self.config.hp.head.model_type == 'gmm':
        #     gen_logits = self.model.head(feats.detach())
        #     logits_for_true = gen_logits[torch.arange(gen_logits.size(0)), y] # [batch_size]
        #     gen_loss = logits_for_true.mean()

        # loss += 0.1 * gen_loss

        # if self.config.hp.get('push_protos_apart_loss_coef', 0.0) > 0:
        #     mean_distance = compute_mean_distance(protos)
        #     loss += self.config.hp.push_protos_apart_loss_coef * (-1 * mean_distance)
        #     self.writer.add_scalar('mean_distance', mean_distance.item(), self.num_iters_done)

        # if self.config.hp.get('protos_clf_loss_coef', 0.0) > 0:
        #     protos_clf_targets = torch.arange(protos.size(1)).to(protos.device) # [n_classes]
        #     protos_clf_targets = protos_clf_targets.unsqueeze(1).repeat(1, protos.size(0)) # [n_classes, n_protos]
        #     protos_clf_targets = protos_clf_targets.permute(1, 0) # [n_protos, n_classes]
        #     protos_main = protos.mean(dim=0) # [n_classes, hid_dim]
        #     protos_main = normalize(protos_main, self.model.head.config.scale.value) # [n_classes, hid_dim]
        #     protos_clf_logits = protos @ protos_main.t() # [n_protos, n_classes, n_classes]

        #     protos_clf_loss = F.cross_entropy(protos_clf_logits, protos_clf_targets)
        #     loss += protos_clf_loss * self.config.hp.protos_clf_loss_coef
        #     self.writer.add_scalar('protos_clf_loss', protos_clf_loss.item(), self.num_iters_done)

        # if self.config.hp.get('generative_training.loss_coef'):
        #     generative_loss = self.compute_generative_loss()
        #     loss += self.config.hp.generative_training.loss_coef * generative_loss
        #     self.writer.add_scalar(f'{self.config.hp.generative_training.type}_loss', generative_loss.item(), self.num_iters_done)

        if self.config.hp.get('diagonal_cov_reg.loss_magnitude'):
            diagonal_cov_reg = self.compute_diagonal_cov_reg()
            loss += self.config.hp.diagonal_cov_reg.loss_magnitude * diagonal_cov_reg / diagonal_cov_reg.abs().item()
            self.writer.add_scalar(f'diagonal_cov_reg', diagonal_cov_reg.item(), self.num_iters_done)

        # if self.config.hp.get('reverse_clf.loss_coef'):
        #     reverse_clf_loss = self.compute_reverse_clf_loss()
        #     loss += self.config.hp.reverse_clf.loss_coef * reverse_clf_loss
        #     self.writer.add_scalar(f'reverse_clf_loss', reverse_clf_loss.item(), self.num_iters_done)

        # if self.config.hp.get('fake_clf.loss_coef'):
        #     fake_clf_loss = self.compute_fake_clf_loss()
        #     loss += self.config.hp.fake_clf.loss_coef * fake_clf_loss
        #     self.writer.add_scalar(f'fake_clf_loss', fake_clf_loss.item(), self.num_iters_done)

        # if self.config.hp.get('pull_golden_protos.loss_coef'):
        #     pull_golden_protos_loss = self.compute_pull_golden_protos_loss()
        #     loss += self.config.hp.pull_golden_protos.loss_coef * pull_golden_protos_loss
        #     self.writer.add_scalar(f'pull_golden_protos_loss', pull_golden_protos_loss.item(), self.num_iters_done)

        # if self.config.hp.get('creativity_reg.loss_coef'):
        #     creativity_reg_loss = self.compute_creativity_reg()
        #     loss += self.config.hp.creativity_reg.loss_coef * creativity_reg_loss
        #     self.writer.add_scalar(f'creativity_reg_loss', creativity_reg_loss.item(), self.num_iters_done)

        if self.config.hp.head.get('grad_reg_coef'):
            grad_reg = self.compute_gradient_penalty(loss)
            loss += self.config.hp.head.grad_reg_coef * grad_reg
            self.writer.add_scalar(f'grad_reg', grad_reg.item(), self.num_iters_done)

        self.optim.zero_grad()
        loss.backward()
        if self.config.hp.get('clip_grad.value', float('inf')) < float('inf'):
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.hp.clip_grad.value)
            self.writer.add_scalar('cls/grad_norm', grad_norm, self.num_iters_done)
        self.optim.step()

        self.writer.add_scalar('cls_loss', cls_loss.item(), self.num_iters_done)

    # def compute_creativity_reg(self) -> Tensor:
    #     protos = self.model.head.generate_prototypes()

    def compute_diagonal_cov_reg(self) -> Tensor:
        if not self.config.hp.get('diagonal_cov_reg.loss_coef'): return torch.tensor(0.0)

        batch = self.sample_batch(self.task_ds_train, self.config.hp.diagonal_cov_reg.batch_size)
        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        feats = self.model.embedder(x)
        reg = compute_diagonal_cov_reg(feats)

        return reg

    def compute_rehearsal_loss(self) -> Tensor:
        X, y = self.sample_from_memory(self.config.hp.rehearsal.batch_size)
        logits = prune_logits(self.model(X), self.seen_classes_mask)

        return F.cross_entropy(logits, y)

    # def compute_pull_golden_protos_loss(self) -> Tensor:
    #     protos = self.model.head.generate_prototypes(1, golden=True) # [1, n_classes, hid_dim]
    #     centroids = self.compute_centroids() # [n_task_classes, hid_dim]

    #     protos = normalize(protos, self.config.hp.head.common.scale.value)
    #     centroids = normalize(centroids, self.config.hp.head.common.scale.value)

    #     assert protos.shape == (1, self.config.data.num_classes, self.config.hp.head.common.hid_dim)
    #     assert centroids.shape == (len(self.classes), self.config.hp.head.common.hid_dim)

    #     protos = protos.squeeze(0) # [n_classes, hid_dim]
    #     protos = protos[self.classes] # [n_task_classes, hid_dim]
    #     mean_dist = (protos - centroids).pow(2).mean()

    #     return mean_dist

    # def compute_centroids(self) -> Tensor:
    #     n_task_classes = len(self.classes)
    #     hid_dim = self.config.hp.head.common.hid_dim
    #     n_imgs_per_class = len(self.core_images) // n_task_classes

    #     with torch.no_grad():
    #         feats = self.model.embedder(self.core_images.to(self.device_name)) # [n_task_classes * n_imgs_per_class, hid_dim]
    #         feats = feats.view(n_task_classes, n_imgs_per_class, hid_dim) # [n_task_classes, n_imgs_per_class, hid_dim]
    #         feats = normalize(feats, self.model.head.config.scale.value) # [n_task_classes, n_imgs_per_class, hid_dim]
    #         centroids = feats.mean(dim=1) # [n_task_classes, hid_dim]
    #         centroids = normalize(centroids, self.model.head.config.scale.value) # [n_task_classes, hid_dim]

    #     return centroids

    # def compute_reverse_clf_loss(self) -> Tensor:
    #     prototypes = self.model.head.generate_prototypes() # [n_protos, n_classes, hid_dim]
    #     prototypes = prototypes[:, self.classes, :] # [n_protos, n_task_classes, hid_dim]
    #     prototypes = normalize(prototypes, self.model.head.config.scale.value) # [n_protos, n_task_classes, hid_dim]

    #     n_protos = prototypes.size(0)
    #     n_task_classes = len(self.classes)

    #     centroids = self.compute_centroids()

    #     reverse_logits = prototypes @ centroids.T # [n_protos, n_task_classes, n_task_classes]
    #     reverse_logits = reverse_logits.view(n_protos * n_task_classes, n_task_classes)
    #     targets = torch.arange(n_task_classes).repeat(n_protos).to(self.device_name)

    #     return F.cross_entropy(reverse_logits, targets)

    # def compute_fake_clf_loss(self) -> Tensor:
    #     normal_prototypes = self.model.head.generate_prototypes() # [n_protos, n_classes, hid_dim]

    #     with torch.no_grad():
    #         golden_prototypes = self.model.head.generate_prototypes(1, golden=True) # [1, n_classes, hid_dim]
    #         golden_prototypes = golden_prototypes.squeeze(0)

    #     golden_prototypes = normalize(golden_prototypes, self.model.head.config.scale.value) # [n_classes, hid_dim]
    #     normal_prototypes = normalize(normal_prototypes, self.model.head.config.scale.value) # [n_classes, hid_dim]

    #     n_protos, n_classes, _ = normal_prototypes.shape
    #     fake_logits = normal_prototypes @ golden_prototypes.T # [n_protos, n_classes, n_classes]
    #     fake_logits = fake_logits.view(n_protos * n_classes, n_classes)
    #     targets = torch.arange(n_classes).repeat(n_protos).to(self.device_name) # [n_protos * n_classes]

    #     return F.cross_entropy(fake_logits, targets)

    # def compute_generative_loss(self):
    #     prototypes = self.model.head.generate_prototypes(self.config.hp.generative_training.num_protos) # [n_protos, n_classes, hid_dim]
    #     prototypes = prototypes[:, self.classes, :] # [n_protos, n_curr_classes, hid_dim]
    #     prototypes = prototypes.view(-1, prototypes.size(2)) # [n_protos * n_curr_classes, hid_dim]

    #     batch = self.sample_batch(self.task_ds_train, prototypes.size(0), replace=True)
    #     x = torch.from_numpy(np.array(batch[0])).to(self.device_name)

    #     with torch.no_grad():
    #         feats = self.model.embedder(x) # [batch_size, hid_dim]

    #     if self.config.hp.generative_training.type == 'gdpp':
    #         generative_loss = compute_gdpp_loss(prototypes, feats)
    #     elif self.config.hp.generative_training.type == 'mmd':
    #         generative_loss = compute_mmd_loss(prototypes, feats, self.config.hp.generative_training.cov_diff_coef)
    #     else:
    #         raise NotImplementedError(f'Unknown generative loss type: {self.config.hp.generative_training.type}')

    #     return generative_loss

    # def _after_train_hook(self):
    #     if not self.config.get('logging.save_prototypes'):
    #         return

    #     with torch.no_grad():
    #         prototypes = self.model.head.generate_prototypes()
    #         prototypes = normalize(prototypes, self.model.head.config.scale.value)
    #     filename = os.path.join(self.main_trainer.paths.custom_data_path, f'prototypes-{self.task_idx + 1}')
    #     np.save(filename, prototypes.cpu().numpy())
    # def compute_grad_regularization(self):
    #     if self.task_idx - self.config.get('start_task', 0) <= 0:
    #         return torch.tensor(0.)

    #     lr = self.config.hp.optim.kwargs.lr
    #     W = self.model.head.transform.weight # [hid_dim, attr_dim]
    #     W_next = W - lr * W.grad
    #     old_attrs = torch.from_numpy(self.attrs[self.seen_classes]).to(self.device_name)
    #     old_protos = old_attrs @ W.t() # [n_seen_classes, hid_dim]
    #     new_protos = old_attrs @ W_next.t() # [n_seen_classes, hid_dim]
    #     diff = (old_protos - new_protos).norm(dim=1).mean()

    #     return diff

    def compute_gradient_penalty(self, loss):
        if self.task_idx - self.config.get('start_task', 0) <= 0:
            return torch.tensor(0.)

        old_attrs = torch.from_numpy(self.attrs[self.seen_classes]).to(self.device_name)

        # loss = discriminator(interpolations, y)

        W_grad = autograd.grad(loss, self.model.head.transform.weight, create_graph=True)[0]
        penalty = (old_attrs @ W_grad.t()).norm(dim=1).mean()

        return penalty

