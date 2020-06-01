from typing import List
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from sklearn.model_selection import train_test_split

from src.utils.training_utils import construct_optimizer, prune_logits, normalize
from src.utils.data_utils import construct_output_mask
from src.utils.metrics import remap_targets


class ZSLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        config = config.overwrite(config[config.dataset])
        config = config.overwrite(Config.read_from_cli())
        config.exp_name = f'zsl_{config.dataset}_{config.hp.compute_hash()}_{config.random_seed}'
        if not config.get('silent'):
            print(config.hp)
        self.random = np.random.RandomState(config.random_seed)
        super().__init__(config)

    def init_dataloaders(self):
        feats = np.load(f'{self.config.data.dir}/feats.npy').astype(np.float32)
        # feats = (feats - feats.mean(axis=0, keepdims=True)) / feats.std(axis=0, keepdims=True)
        labels = np.load(f'{self.config.data.dir}/labels.npy').astype(int)
        attrs = np.load(f'{self.config.data.dir}/attrs.npy').astype(np.float32)
        train_idx = np.load(f'{self.config.data.dir}/train_idx.npy')
        test_idx = np.load(f'{self.config.data.dir}/test_idx.npy')

        self.seen_mask = construct_output_mask(self.config.data.seen_classes, self.config.data.num_classes)
        self.attrs = torch.from_numpy(attrs).to(self.device_name)
        self.attrs_seen = self.attrs[self.seen_mask]
        self.test_labels = np.array([labels[i] for i in test_idx])
        self.test_seen_idx = [i for i, y in enumerate(self.test_labels) if y in self.config.data.seen_classes]
        self.test_unseen_idx = [i for i, y in enumerate(self.test_labels) if y in self.config.data.unseen_classes]

        seen_classes = list(sorted(list(self.config.data.seen_classes)))
        val_remapped_labels = np.array(remap_targets(labels, seen_classes))

        # Allocating a portion of the train set for cross-validation
        if self.config.hp.val_ratio > 0:
            num_train_classes = int(len(seen_classes) * (1 - self.config.hp.val_ratio))
            self.train_classes = self.random.choice(seen_classes, size=num_train_classes, replace=False)
            self.train_classes = sorted(self.train_classes)
            self.val_classes = [c for c in seen_classes if not c in self.train_classes]

            val_pseudo_unseen_idx = np.array([i for i, c in enumerate(labels) if c in self.val_classes])
            train_idx = np.array([i for i, c in enumerate(labels) if c in self.train_classes])
            train_remapped_labels = np.array(remap_targets(labels, self.train_classes))

            # Additionally extend seen_val_idx with "seen seen" data
            # seen_seen_val_idx = seen_train_idx[self.random.choice(seen_train_idx, size=)]
            train_idx, val_pseudo_seen_idx = train_test_split(train_idx, test_size=self.config.hp.val_ratio)
            val_idx = np.hstack([val_pseudo_seen_idx, val_pseudo_unseen_idx])
            self.val_pseudo_seen_idx = np.arange(len(val_pseudo_seen_idx))
            self.val_pseudo_unseen_idx = len(val_pseudo_seen_idx) + np.arange(len(val_pseudo_unseen_idx))
            self.val_labels = np.array([val_remapped_labels[i] for i in val_idx])
            self.val_scope = 'seen'

            ds_train = [(feats[i], train_remapped_labels[i]) for i in train_idx]
            ds_val = [(feats[i], val_remapped_labels[i]) for i in val_idx]
            ds_test = [(feats[i], labels[i]) for i in test_idx]

            assert np.all(np.array(train_remapped_labels)[train_idx] >= 0)
            assert np.all(np.array(val_remapped_labels)[val_idx] >= 0)
        else:
            self.logger.warn('Running without validation!')
            # We are doing the final run, so let's use all the data for training
            self.train_classes = sorted(seen_classes)
            train_remapped_labels = np.array(remap_targets(labels, self.train_classes))
            self.val_pseudo_seen_idx = self.test_seen_idx
            self.val_pseudo_unseen_idx = self.test_unseen_idx
            self.val_labels = self.test_labels
            self.val_scope = 'all'

            ds_train = [(feats[i], train_remapped_labels[i]) for i in train_idx]
            ds_test = [(feats[i], labels[i]) for i in test_idx]
            ds_val = ds_test

        self.train_dataloader = DataLoader(ds_train, batch_size=self.config.hp.batch_size, shuffle=True, num_workers=0)
        self.val_dataloader = DataLoader(ds_val, batch_size=2048, num_workers=0)
        self.test_dataloader = DataLoader(ds_test, batch_size=2048, num_workers=0)
        self.train_seen_mask = construct_output_mask(self.train_classes, self.config.data.num_classes)

        self.curr_val_scores = [0, 0, 0]
        self.best_val_scores = [0, 0, 0]
        self.test_scores = [0, 0, 0]

    def _run_training(self):
        start_time = time()

        for epoch in range(1, self.config.hp.max_num_epochs + 1):
            for batch in self.train_dataloader:
                self.train_on_batch(batch)
                self.num_iters_done += 1

            self.num_epochs_done += 1

            if epoch % self.config.val_freq_epochs == 0:
                self.curr_val_scores = self.validate()
                if not self.config.get('silent'):
                    self.print_scores(self.curr_val_scores)

            self.scheduler.step()

        if self.config.hp.val_ratio > 0:
            if not self.config.get('silent'):
                self.logger.info('<===== Test scores (computed for the highest val scores) =====>')
                self.print_scores(self.test_scores)

        print(f'Training took time: {time() - start_time: .02f} seconds')

    def train_on_batch(self, batch):
        self.model.train()
        feats = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        labels = torch.from_numpy(np.array(batch[1])).to(self.device_name)
        logits = self.compute_logits(feats, scope='train')

        if self.config.hp.get('label_smoothing', 1.0) < 1.0:
            n_classes = logits.shape[1]
            other_prob_val = (1 - self.config.hp.label_smoothing) / n_classes
            targets = torch.ones_like(logits) * other_prob_val
            targets.scatter_(1, labels.unsqueeze(1), self.config.hp.label_smoothing)

            log_probs = logits.log_softmax(dim=1)
            loss = F.kl_div(log_probs, targets)
        else:
            loss = F.cross_entropy(logits, labels)

        if self.config.hp.get('entropy_reg_coef', 0) > 0:
            loss -= self.config.hp.entropy_reg_coef * self.compute_entropy_reg(logits)

        if self.config.hp.get('cross_entropy_reg_coef', 0) > 0:
            loss -= self.config.hp.cross_entropy_reg_coef * self.compute_cross_entropy_reg(logits)

        self.optim.zero_grad()
        loss.backward()
        if self.config.hp.get('grad_clip_val', 0) > 0:
            clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_val)
        self.optim.step()

    def compute_logits(self, feats, scope: str='none'):
        if self.config.hp.get('attrs_dropout.p') and self.model.training:
            mask = (torch.rand(self.attrs.shape[1]) > self.config.hp.attrs_dropout.p)
            mask = mask.unsqueeze(0).float().to(self.attrs.device)
            attrs = self.attrs * mask
            attrs = attrs / (1 - self.config.hp.attrs_dropout.p)
        else:
            attrs = self.attrs

        if self.config.hp.get('feats_dropout.p') and self.model.training:
            feats = F.dropout(feats, p=self.config.hp.feats_dropout.p)

        if scope == 'train':
            # Otherwise unseen classes will leak through batch norm
            attrs = self.attrs[self.train_seen_mask]
        elif scope == 'seen':
            attrs = self.attrs[self.seen_mask]
        elif scope == 'none':
            attrs = self.attrs

        protos = self.model(attrs)
        feats = normalize(feats, scale_value=self.config.hp.scale)
        protos = normalize(protos, scale_value=self.config.hp.scale)
        logits = feats @ protos.t()

        return logits

    def compute_entropy_reg(self, logits):
        log_probs = logits.log_softmax(dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)

        return entropy.mean()

    def compute_cross_entropy_reg(self, logits):
        return logits.log_softmax(dim=1).mean()

    def init_models(self):
        output_layer = nn.Linear(self.config.hp.hid_dim, self.config.hp.feat_dim)
        bn_layer = nn.BatchNorm1d(self.config.hp.hid_dim, affine=self.config.hp.bn_affine)

        self.model = nn.Sequential(
            nn.Linear(self.attrs.shape[1], self.config.hp.hid_dim),
            nn.ReLU(),
            bn_layer,
            output_layer,
            nn.ReLU()
        ).to(self.device_name)

        if self.config.hp.init.type == 'proper':
            if self.config.hp.init.with_relu:
                var = 2 / (self.config.hp.hid_dim * self.config.hp.feat_dim * (1 - 1/np.pi))
            else:
                var = 1 / (self.config.hp.hid_dim * self.config.hp.feat_dim)

            if self.config.hp.init.dist == 'uniform':
                b = np.sqrt(3 * var)
                output_layer.weight.data.uniform_(-b, b)
            else:
                output_layer.weight.data.normal_(0, np.sqrt(var))

    def init_optimizers(self):
        self.optim = construct_optimizer(self.model.parameters(), self.config.hp.optim)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=self.config.hp.optim.scheduler.step_size, gamma=self.config.hp.optim.scheduler.gamma)

    def run_inference(self, dataloader: DataLoader, scope: str='all'):
        with torch.no_grad():
            logits = [self.compute_logits(x.to(self.device_name), scope).cpu() for x, _ in dataloader]
        logits = torch.cat(logits, dim=0)

        return logits

    def compute_scores(self, dataset: str='val'):
        self.model.eval()

        if dataset == 'val':
            logits = self.run_inference(self.val_dataloader, scope=self.val_scope)
            preds = logits.argmax(dim=1).numpy()
            guessed = (preds == self.val_labels)
            seen_acc = guessed[self.val_pseudo_seen_idx].mean()
            unseen_acc = guessed[self.val_pseudo_unseen_idx].mean()
            harmonic = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)
        elif dataset == 'test':
            logits = self.run_inference(self.test_dataloader, scope='all')
            preds = logits.argmax(dim=1).numpy()
            guessed = (preds == self.test_labels)
            seen_acc = guessed[self.test_seen_idx].mean()
            unseen_acc = guessed[self.test_unseen_idx].mean()
            harmonic = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)
        else:
            raise ValueError(f"Wrong dataset for GZSL scores: {dataset}")

        return 100 * np.array([seen_acc, unseen_acc, harmonic])

    def validate(self):
        scores = self.compute_scores(dataset='val')

        if scores[2] >= self.best_val_scores[2]:
            self.best_val_scores = scores

            # Compute test scores but keep it hidden
            self.test_scores = self.compute_scores(dataset='test')
            self.logger.info(f'[TEST SCORES] GZSL-S: {self.test_scores[0]: .4f}. GZSL-U: {self.test_scores[1]: .4f}. GZSL-H: {self.test_scores[2]: .4f}')

        return scores

    def print_scores(self, scores: List[float]):
        self.logger.info(f'[Epoch #{self.num_epochs_done: 3d}] GZSL-S: {scores[0]: .4f}. GZSL-U: {scores[1]: .4f}. GZSL-H: {scores[2]: .4f}')
