from typing import List
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as init
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from sklearn.model_selection import train_test_split

from src.utils.training_utils import construct_optimizer, normalize, prune_logits
from src.utils.data_utils import construct_output_mask
from src.utils.metrics import remap_targets, compute_ausuc


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

        if self.config.hp.standardize_attrs:
            attrs = (attrs - attrs.mean(axis=0, keepdims=True)) / attrs.std(axis=0, keepdims=True)

        self.seen_classes = list(sorted(list(self.config.data.seen_classes)))
        self.unseen_classes = list(sorted(list(self.config.data.unseen_classes)))
        self.seen_mask = construct_output_mask(self.seen_classes, self.config.data.num_classes)
        self.unseen_mask = construct_output_mask(self.unseen_classes, self.config.data.num_classes)
        self.attrs = torch.from_numpy(attrs).to(self.device_name)
        self.attrs_seen = self.attrs[self.seen_mask]
        self.test_labels = np.array([labels[i] for i in test_idx])
        self.test_seen_idx = [i for i, y in enumerate(self.test_labels) if y in self.seen_classes]
        self.test_unseen_idx = [i for i, y in enumerate(self.test_labels) if y in self.unseen_classes]
        self.remapped_unseen_test_labels = remap_targets(labels[self.test_unseen_idx], self.unseen_classes)

        # Allocating a portion of the train set for cross-validation
        if self.config.hp.val_ratio > 0:
            num_train_classes = int(len(self.seen_classes) * (1 - self.config.hp.val_ratio))
            self.train_classes = self.random.choice(self.seen_classes, size=num_train_classes, replace=False)
            self.train_classes = sorted(self.train_classes)
            self.pseudo_unseen_classes = [c for c in self.seen_classes if not c in self.train_classes]

            val_pseudo_unseen_idx = np.array([i for i, c in enumerate(labels) if c in self.pseudo_unseen_classes])
            train_idx = np.array([i for i, c in enumerate(labels) if c in self.train_classes])
            train_remapped_labels = np.array(remap_targets(labels, self.train_classes))

            # Additionally extend seen_val_idx with "seen seen" data
            # seen_seen_val_idx = seen_train_idx[self.random.choice(seen_train_idx, size=)]
            train_idx, val_pseudo_seen_idx = train_test_split(train_idx, test_size=self.config.hp.val_ratio)
            val_idx = np.hstack([val_pseudo_seen_idx, val_pseudo_unseen_idx])
            self.val_pseudo_seen_idx = np.arange(len(val_pseudo_seen_idx))
            self.val_pseudo_unseen_idx = len(val_pseudo_seen_idx) + np.arange(len(val_pseudo_unseen_idx))
            val_remapped_labels = np.array(remap_targets(labels, self.seen_classes))
            self.val_labels = np.array([val_remapped_labels[i] for i in val_idx])
            self.val_scope = 'seen'
            self.pseudo_unseen_mask = construct_output_mask(
                remap_targets(self.pseudo_unseen_classes, self.seen_classes), len(self.seen_classes))

            ds_train = [(feats[i], train_remapped_labels[i]) for i in train_idx]
            ds_val = [(feats[i], val_remapped_labels[i]) for i in val_idx]
            ds_test = [(feats[i], labels[i]) for i in test_idx]

            assert np.all(np.array(train_remapped_labels)[train_idx] >= 0)
            assert np.all(np.array(val_remapped_labels)[val_idx] >= 0)
        else:
            self.logger.warn('Running without validation!')
            # We are doing the final run, so let's use all the data for training
            self.train_classes = self.seen_classes
            train_remapped_labels = np.array(remap_targets(labels, self.train_classes))
            self.val_pseudo_seen_idx = self.test_seen_idx
            self.val_pseudo_unseen_idx = self.test_unseen_idx
            self.pseudo_unseen_classes = self.unseen_classes
            self.pseudo_unseen_mask = self.unseen_mask
            self.val_labels = self.test_labels
            self.val_scope = 'all'

            ds_train = [(feats[i], train_remapped_labels[i]) for i in train_idx]
            ds_test = [(feats[i], labels[i]) for i in test_idx]
            ds_val = ds_test

        self.train_dataloader = DataLoader(ds_train, batch_size=self.config.hp.batch_size, shuffle=True, num_workers=0)
        self.val_dataloader = DataLoader(ds_val, batch_size=2048, num_workers=0)
        self.test_dataloader = DataLoader(ds_test, batch_size=2048, num_workers=0)
        self.train_seen_mask = construct_output_mask(self.train_classes, self.config.data.num_classes)

        self.curr_val_scores = [0, 0, 0, 0]
        self.best_val_scores = [0, 0, 0, 0]
        self.test_scores = [0, 0, 0, 0]

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
                    self.print_scores(self.curr_val_scores, prefix='[CURR VAL] ')

            self.scheduler.step()

        self.print_scores(self.test_scores, prefix='[TEST] ')
        self.print_scores(self.curr_val_scores, prefix='[FINAL VAL] ')

        self.elapsed = time() - start_time
        print(f'Training took time: {self.elapsed: .02f} seconds')

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
        elif scope == 'unseen':
            attrs = self.attrs[self.unseen_mask]

        protos = self.model(attrs)
        if self.config.hp.normalize_and_scale:
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
        if self.config.hp.model_type == 'deep':
            penultimate_dim = self.config.hp.hid_dim
            early_layers = nn.Sequential(
                nn.Linear(self.attrs.shape[1], penultimate_dim),
                nn.ReLU(),
            )
            final_activation = nn.ReLU()
        else:
            penultimate_dim = self.attrs.shape[1]
            early_layers = nn.Identity()
            final_activation = nn.Identity()

        if self.config.hp.has_bn:
            bn_layer = nn.BatchNorm1d(penultimate_dim, affine=self.config.hp.get('bn_affine', False))
        else:
            bn_layer = nn.Identity()

        if self.config.hp.has_dn:
            dn_layer = DynamicNormalization()
        else:
            dn_layer = nn.Identity()

        output_layer = nn.Linear(penultimate_dim, self.config.hp.feat_dim)

        self.model = nn.Sequential(
            early_layers,
            bn_layer,
            dn_layer,
            output_layer,
            final_activation
        ).to(self.device_name)

        if self.config.hp.init.type == 'proper':
            if self.config.hp.init.with_relu:
                var = 2 / (penultimate_dim * self.config.hp.feat_dim * (1 - 1/np.pi))
            else:
                var = 1 / (penultimate_dim * self.config.hp.feat_dim)

            if self.config.hp.init.dist == 'uniform':
                b = np.sqrt(3 * var)
                output_layer.weight.data.uniform_(-b, b)
            else:
                output_layer.weight.data.normal_(0, np.sqrt(var))
        elif self.config.hp.init.type == 'xavier':
            init.xavier_uniform_(output_layer.weight)
        elif self.config.hp.init.type == 'kaiming':
            init.kaiming_uniform_(output_layer.weight, mode=self.config.hp.init.mode, nonlinearity='relu')
        else:
            raise ValueError(f'Unknown init type: {self.config.hp.init.type}')

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
            # GZSL metrics
            logits = self.run_inference(self.val_dataloader, scope=self.val_scope)
            preds = logits.argmax(dim=1).numpy()
            guessed = (preds == self.val_labels)
            seen_acc = guessed[self.val_pseudo_seen_idx].mean()
            unseen_acc = guessed[self.val_pseudo_unseen_idx].mean()
            harmonic = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

            # ZSL
            zsl_logits = prune_logits(logits, self.pseudo_unseen_mask)
            zsl_preds = zsl_logits.argmax(dim=1).numpy()
            zsl_acc = (zsl_preds == self.val_labels)[self.val_pseudo_unseen_idx].mean()

            # AUSUC
            if self.config.get('logging.compute_ausuc'):
                ausuc = compute_ausuc(logits, self.val_labels, self.train_seen_mask) * 0.01
            else:
                ausuc = 0
        elif dataset == 'test':
            # GZSL metrics
            logits = self.run_inference(self.test_dataloader, scope='all')
            preds = logits.argmax(dim=1).numpy()
            guessed = (preds == self.test_labels)
            seen_acc = guessed[self.test_seen_idx].mean()
            unseen_acc = guessed[self.test_unseen_idx].mean()
            harmonic = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

            # ZSL
            zsl_logits = prune_logits(logits, self.unseen_mask)
            zsl_preds = zsl_logits.argmax(dim=1).numpy()
            zsl_acc = (zsl_preds == self.test_labels)[self.test_unseen_idx].mean()

            # AUSUC
            if self.config.get('logging.compute_ausuc'):
                ausuc = compute_ausuc(logits, self.test_labels, self.seen_mask) * 0.01
            else:
                ausuc = 0
        else:
            raise ValueError(f"Wrong dataset for GZSL scores: {dataset}")

        return 100 * np.array([seen_acc, unseen_acc, harmonic, zsl_acc, ausuc])

    def validate(self):
        scores = self.compute_scores(dataset='val')

        if scores[2] >= self.best_val_scores[2]:
            self.best_val_scores = scores

            # Compute test scores but keep it hidden
            self.test_scores = self.compute_scores(dataset='test')

            if not self.config.get('silent'):
                self.print_scores(self.test_scores, prefix='[TEST] ')

        return scores

    def print_scores(self, scores: List[float], prefix=''):
        self.logger.info(
            f'{prefix}[Epoch #{self.num_epochs_done: 3d}] ' \
            f'GZSL-S: {scores[0]:.4f}. ' \
            f'GZSL-U: {scores[1]:.4f}. ' \
            f'GZSL-H: {scores[2]:.4f}. ' \
            f'ZSL: {scores[3]:.4f}. ' \
            f'AUSUC: {scores[4]:.4f}.')


class DynamicNormalization(nn.Module):
    def forward(self, x):
        assert x.ndim == 2, f"Wrong shape: {x.shape}"

        mean_norm = x.norm(dim=1).mean()
        return x / mean_norm.pow(2)

