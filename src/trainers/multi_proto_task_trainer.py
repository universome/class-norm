import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits, normalize
from src.utils.losses import compute_mean_distance


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
            feats_rec = self.model.head.model.compute_dae_reconstructions(feats, y)

            rec_loss = torch.norm(feats_rec - feats, dim=1).mean()
            loss += self.config.hp.head.dae.loss_coef * rec_loss
            self.writer.add_scalar('rec_loss', rec_loss.item(), self.num_iters_done)

        if self.config.hp.head.get('protos_clf_loss.enabled') or self.config.hp.head.get('push_protos_apart.enabled'):
            logits, protos = self.model(x, return_protos=True)
        else:
            logits = self.model(x)

        if self.config.hp.head.aggregation_type == 'aggregate_losses':
            n_protos = logits.size(0) // y.size(0)
            batch_size = y.size(0)
            y = y.view(batch_size, 1).repeat(1, n_protos).view(batch_size * n_protos)
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)
        else:
            cls_loss = self.criterion(prune_logits(logits, self.output_mask), y)

        loss += cls_loss

        if self.config.hp.head.get('push_protos_apart.enabled'):
            mean_distance = compute_mean_distance(protos)
            loss += self.config.hp.head.push_protos_apart.loss_coef * (-1 * mean_distance)

            self.writer.add_scalar('mean_distance', mean_distance.item(), self.num_iters_done)

        if self.config.hp.head.get('protos_clf_loss.enabled'):
            protos_clf_targets = torch.arange(protos.size(1)).to(protos.device) # [n_classes]
            protos_clf_targets = protos_clf_targets.unsqueeze(1).repeat(1, protos.size(0)) # [n_classes, n_protos]
            protos_clf_targets = protos_clf_targets.permute(1, 0) # [n_protos, n_classes]
            protos_main = protos.mean(dim=0) # [n_classes, hid_dim]
            protos_main = normalize(protos_main, self.config.hp.head.scale.value) # [n_classes, hid_dim]
            protos_clf_logits = protos @ protos_main.t() # [n_protos, n_classes, n_classes]

            protos_clf_loss = F.cross_entropy(protos_clf_logits, protos_clf_targets)
            loss += protos_clf_loss * self.config.hp.head.protos_clf_loss.loss_coef
            self.writer.add_scalar('protos_clf_loss', protos_clf_loss.item(), self.num_iters_done)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('cls_loss', cls_loss.item(), self.num_iters_done)
