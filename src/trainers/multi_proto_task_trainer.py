import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits, normalize
from src.utils.losses import compute_mean_distance, compute_gdpp_loss, compute_mmd_loss


class MultiProtoTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        loss = 0.
        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        if self.config.hp.head.get('dae.enabled'):
            with torch.no_grad():
                feats = self.model.embedder(x)
            logits = self.model.head(feats)
            feats_rec = self.model.head.compute_dae_reconstructions(feats, y)

            rec_loss = torch.norm(feats_rec - feats, dim=1).mean()
            loss += self.config.hp.head.dae.loss_coef * rec_loss
            self.writer.add_scalar('rec_loss', rec_loss.item(), self.num_iters_done)

        if self.config.hp.get('triplet_loss.enabled'):
            feats = self.model.embedder(x) # [batch_size, hid_dim]
            protos = self.model.head.generate_prototypes() # [n_protos, n_classes, hid_dim]

            feats = normalize(feats) # [batch_size, hid_dim]
            protos = normalize(protos)

            batch_size = feats.size(0)
            n_protos, _, hid_dim = protos.size()
            n_curr_classes = len(self.classes)

            protos = protos[:, self.classes, :] # [n_protos, n_curr_classes, hid_dim]
            protos = protos.view(n_protos * n_curr_classes, hid_dim).permute(1, 0) # [hid_dim, n_protos * n_curr_classes]
            distances = (feats.unsqueeze(2) - protos.unsqueeze(0)).norm(dim=1).pow(2) # [batch_size, n_protos * n_curr_classes]
            distances = distances.view(batch_size, n_protos, n_curr_classes)
            classes_in_batch = y.view(batch_size, 1, 1).repeat(1, n_protos, n_curr_classes)
            classes_generated = torch.tensor(self.classes).to(self.device_name).view(1, 1, n_curr_classes).repeat(batch_size, n_protos, 1)
            positive_distances = distances.masked_select(classes_in_batch == classes_generated)
            negative_distances = distances.masked_select(classes_in_batch != classes_generated)

            # Now we should remove too easy negative distances
            smallest_negative_distances_idx = negative_distances.sort()[1][:len(positive_distances)]
            negative_distances = negative_distances[smallest_negative_distances_idx]
            #triplet_loss = torch.max(positive_distances - negative_distances + self.config.hp.triplet_loss.margin, 0)
            # TODO: well, this is actually not a triplet loss...
            triplet_loss = positive_distances.mean() - negative_distances.mean()

            loss += triplet_loss * self.config.hp.triplet_loss.coef
            # print('triplet', triplet_loss.item())

            self.writer.add_scalar('triplet_loss/loss', triplet_loss.item(), self.num_iters_done)
            self.writer.add_scalar('triplet_loss/positive_mean_dist', positive_distances.mean().item(), self.num_iters_done)
            self.writer.add_scalar('triplet_loss/negative_mean_dist', negative_distances.mean().item(), self.num_iters_done)

        if self.config.hp.get('protos_clf_loss_coef') or self.config.hp.get('push_protos_apart_loss_coef'):
            logits, protos = self.model(x, return_protos=True)
        else:
            logits = self.model(x)

        if self.config.hp.head.aggregation_type == 'individual_losses':
            n_protos = logits.size(0) // y.size(0)
            batch_size = y.size(0)
            y = y.view(batch_size, 1).repeat(1, n_protos).view(batch_size * n_protos)
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)
        else:
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)

        if self.config.get('cls_loss.enabled', True):
            loss += cls_loss * self.config.get('cls_loss.coef', 1.0)
            # print('cls', cls_loss.item())

        if self.config.hp.get('push_protos_apart_loss_coef', 0.0) > 0:
            mean_distance = compute_mean_distance(protos)
            loss += self.config.hp.push_protos_apart_loss_coef * (-1 * mean_distance)

            self.writer.add_scalar('mean_distance', mean_distance.item(), self.num_iters_done)

        if self.config.hp.get('protos_clf_loss_coef', 0.0) > 0:
            protos_clf_targets = torch.arange(protos.size(1)).to(protos.device) # [n_classes]
            protos_clf_targets = protos_clf_targets.unsqueeze(1).repeat(1, protos.size(0)) # [n_classes, n_protos]
            protos_clf_targets = protos_clf_targets.permute(1, 0) # [n_protos, n_classes]
            protos_main = protos.mean(dim=0) # [n_classes, hid_dim]
            protos_main = normalize(protos_main, self.config.hp.head.scale.value) # [n_classes, hid_dim]
            protos_clf_logits = protos @ protos_main.t() # [n_protos, n_classes, n_classes]

            protos_clf_loss = F.cross_entropy(protos_clf_logits, protos_clf_targets)
            loss += protos_clf_loss * self.config.hp.protos_clf_loss_coef
            self.writer.add_scalar('protos_clf_loss', protos_clf_loss.item(), self.num_iters_done)

        self.optim.zero_grad()
        loss.backward()
        if self.config.hp.get('clip_grad.value', float('inf')) < float('inf'):
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.hp.clip_grad.value)
            self.writer.add_scalar('cls/grad_norm', grad_norm, self.num_iters_done)
        self.optim.step()

        self.writer.add_scalar('cls_loss', cls_loss.item(), self.num_iters_done)

        if self.config.hp.get('generative_training.loss_coef'):
            self.run_generative_training_step()

    def run_generative_training_step(self):
        prototypes = self.model.head.model.generate_prototypes(self.config.hp.generative_training.num_protos) # [n_protos, n_classes, hid_dim]
        prototypes = prototypes[:, self.classes, :] # [n_protos, n_curr_classes, hid_dim]
        prototypes = prototypes.view(-1, prototypes.size(2)) # [n_protos * n_curr_classes, hid_dim]

        batch = self.sample_batch(self.task_ds_train, prototypes.size(0), replace=True)
        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)

        with torch.no_grad():
            feats = self.model.embedder(x) # [batch_size, hid_dim]

        if self.config.hp.generative_training.type == 'gdpp':
            generative_loss = compute_gdpp_loss(prototypes, feats)
        elif self.config.hp.generative_training.type == 'mmd':
            generative_loss = compute_mmd_loss(prototypes, feats)
        else:
            raise NotImplementedError(f'Unknown generative loss type: {self.config.hp.generative_training.type}')

        self.writer.add_scalar(f'{self.config.hp.generative_training.type}_loss', generative_loss.item(), self.num_iters_done)
        generative_loss *= self.config.hp.generative_training.loss_coef

        self.optim.zero_grad()
        generative_loss.backward()
        self.optim.step()
