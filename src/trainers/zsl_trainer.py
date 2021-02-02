import os
from typing import List
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as init
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from firelab.utils.training_utils import fix_random_seed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.training_utils import construct_optimizer, normalize, prune_logits
from src.utils.data_utils import construct_output_mask, remap_targets
from src.utils.metrics import compute_ausuc
from src.models.attrs_head import AttrsHead


class ZSLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        fix_random_seed(config.random_seed, True, True)
        config = config.overwrite(config[config.dataset])
        config = config.overwrite(Config.read_from_cli())
        config.exp_name = f'zsl_{config.dataset}_{config.hp.compute_hash()}_{config.random_seed}'
        if not config.get('silent'):
            print(config.hp)
        self.random = np.random.RandomState(config.random_seed)
        super().__init__(config)

        self.prelogits_mean_history = []
        self.prelogits_std_history = []
        self.grads_info_history = {'input': [], 'output': []}

    def init_dataloaders(self):
        feats = np.load(f'{self.config.data.dir}/feats.npy').astype(np.float32)
        if self.config.hp.standardize_feats:
            feats = (feats - feats.mean(axis=0, keepdims=True)) / feats.std(axis=0, keepdims=True)
        labels = np.load(f'{self.config.data.dir}/labels.npy').astype(int)
        attrs = np.load(f'{self.config.data.dir}/attrs.npy').astype(np.float32)
        train_idx = np.load(f'{self.config.data.dir}/train_idx.npy')
        test_idx = np.load(f'{self.config.data.dir}/test_idx.npy')

        self.seen_classes = list(sorted(list(self.config.data.seen_classes)))
        self.unseen_classes = list(sorted(list(self.config.data.unseen_classes)))
        self.seen_mask = construct_output_mask(self.seen_classes, self.config.data.num_classes)
        self.unseen_mask = construct_output_mask(self.unseen_classes, self.config.data.num_classes)
        if self.config.hp.get('renormalize_unseen'):
            # attrs[self.unseen_classes] = (attrs[self.unseen_classes] / (attrs[self.unseen_classes].mean(axis=0, keepdims=True) + 1e-8)) * attrs[self.seen_classes].mean(axis=0, keepdims=True)
            attrs[self.unseen_classes] = (attrs[self.unseen_classes] - (attrs[self.unseen_classes].mean(axis=0, keepdims=True) + 1e-8)) / (attrs[self.unseen_classes].std(axis=0, keepdims=True) + 1e-8)
            attrs[self.unseen_classes] = attrs[self.unseen_classes] * attrs[self.seen_classes].std(axis=0, keepdims=True) + attrs[self.seen_classes].mean(axis=0, keepdims=True)

        self.attrs = torch.from_numpy(attrs).to(self.device_name)
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

            self.ds_train = [(feats[i], train_remapped_labels[i]) for i in train_idx]
            self.ds_val = [(feats[i], val_remapped_labels[i]) for i in val_idx]
            self.ds_test = [(feats[i], labels[i]) for i in test_idx]

            assert np.all(np.array(train_remapped_labels)[train_idx] >= 0)
            assert np.all(np.array(val_remapped_labels)[val_idx] >= 0)
        else:
            if not self.config.get('silent'):
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
            self.class_indices_inside_test = {c: [i for i in range(len(test_idx)) if labels[test_idx[i]] == c] for c in range(self.config.data.num_classes)}

            self.ds_train = [(feats[i], train_remapped_labels[i]) for i in train_idx]
            self.ds_test = [(feats[i], labels[i]) for i in test_idx]
            self.ds_val = self.ds_test

        self.train_dataloader = DataLoader(self.ds_train, batch_size=self.config.hp.batch_size, shuffle=True, num_workers=0)
        self.val_dataloader = DataLoader(self.ds_val, batch_size=2048, num_workers=0)
        self.test_dataloader = DataLoader(self.ds_test, batch_size=2048, num_workers=0)
        self.train_seen_mask = construct_output_mask(self.train_classes, self.config.data.num_classes)

        self.curr_val_scores = [0, 0, 0, 0]
        self.best_val_scores = [0, 0, 0, 0]
        self.test_scores = [0, 0, 0, 0]

    def _run_training(self):
        start_time = time()

        for epoch in range(1, self.config.hp.max_num_epochs + 1):
            for batch in self.train_dataloader:
                if self.config.logging.save_grads.freq > 0 and self.num_iters_done % self.config.logging.save_grads.freq == 0:
                    self.compute_grads()

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
        if not self.config.get('silent'):
            self.logger.info(f'Training took time: {self.elapsed: .02f} seconds')

        if self.config.get('save_checkpoint'):
            torch.save(self.model.state_dict(), f'models/checkpoint-{self.config.dataset}-{self.config.hp.compute_hash()}.pt')

        if self.config.logging.compute_prelogits_stats:
            np.save(os.path.join(self.paths.custom_data_path, 'prelogits_mean_history'), self.prelogits_mean_history)
            np.save(os.path.join(self.paths.custom_data_path, 'prelogits_std_history'), self.prelogits_std_history)

        if self.config.logging.save_grads.freq > 0:
            np.savez(os.path.join(self.paths.custom_data_path, 'grads_info_history'),
                input=self.grads_info_history['input'], output=self.grads_info_history['output'])

    def train_on_batch(self, batch):
        self.model.train()
        feats = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        labels = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        if self.config.logging.compute_prelogits_stats \
        or (self.config.logging.save_init_prelogits and self.num_iters_done == 0):
            logits, prelogits = self.compute_logits(feats, scope='train', return_prelogits=True)

            if self.config.logging.compute_prelogits_stats:
                self.prelogits_mean_history.append(prelogits.mean().cpu().item())
                self.prelogits_std_history.append(prelogits.std().cpu().item())

            if self.config.logging.save_init_prelogits and self.num_iters_done == 0:
                np.save(os.path.join(self.paths.custom_data_path, 'prelogits_initial'), prelogits.detach().cpu().numpy())
        else:
            logits = self.compute_logits(feats, scope='train')

        if self.config.hp.get('label_smoothing', 1.0) < 1.0:
            n_classes = logits.shape[1]
            other_prob_val = (1 - self.config.hp.label_smoothing) / n_classes
            targets = torch.ones_like(logits) * other_prob_val
            targets.scatter_(1, labels.unsqueeze(1), self.config.hp.label_smoothing)

            log_probs = logits.log_softmax(dim=1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean')
        else:
            loss = F.cross_entropy(logits, labels)

        # if self.config.hp.get('entropy_reg_coef', 0) > 0:
        #     loss -= self.config.hp.entropy_reg_coef * self.compute_entropy_reg(logits)

        # if self.config.hp.get('cross_entropy_reg_coef', 0) > 0:
        #     loss -= self.config.hp.cross_entropy_reg_coef * self.compute_cross_entropy_reg(logits)

        self.optim.zero_grad()
        loss.backward()
        if self.config.hp.get('grad_clip_val', 0) > 0:
            norm_type = 2 if self.config.hp.grad_clip_norm_type == 'l2' else 'inf'
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_val, norm_type)
        self.optim.step()

    def compute_grads(self):
        self.model.train()

        self.optim.zero_grad()
        input_grads = []
        output_grads = []
        dataloader = DataLoader(
            self.ds_train[:self.config.logging.save_grads.num_points],
            batch_size=self.config.logging.save_grads.batch_size, shuffle=False, num_workers=0)

        for batch in tqdm(dataloader, desc='Computing grads'):
            feats = torch.from_numpy(np.array(batch[0])).to(self.device_name)
            labels = torch.from_numpy(np.array(batch[1])).to(self.device_name)
            logits = self.compute_logits(feats, scope='train')
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            input_grads.append(self.model.early_layers[0].weight.grad.data.cpu())
            output_grads.append(self.model.output_layer.weight.grad.data.cpu())
            self.optim.zero_grad()

        self.grads_info_history['input'].append(self.get_grads_stats(input_grads))
        self.grads_info_history['output'].append(self.get_grads_stats(output_grads))

    @torch.no_grad()
    def get_grads_stats(self, grads: List[Tensor]) -> List[float]:
        grads = torch.stack(grads) # [num_points, z_dim, h_dim]
        grads_mean = grads.mean(dim=0).flatten().norm() # signal
        grads_std = grads.std(dim=0).flatten().norm() # signal
        grads_snr = grads_mean / grads_std
        grads_avg_var = grads.var(dim=0).mean()

        return [grads_mean.item(), grads_std.item(), grads_snr.item(), grads_avg_var.item()]

    def compute_entropy_reg(self, logits):
        log_probs = logits.log_softmax(dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)

        return entropy.mean()

    def compute_cross_entropy_reg(self, logits):
        return logits.log_softmax(dim=1).mean()

    def init_models(self):
        self.model = AttrsHead(self.config.hp.model, self.attrs.cpu().numpy()).to(self.device_name)
        # for n, p in self.model.named_parameters():
        #     print(n, p.mean().item(), p.std().item())

    def init_optimizers(self):
        self.optim = construct_optimizer(self.model.parameters(), self.config.hp.optim)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=self.config.hp.optim.scheduler.step_size, gamma=self.config.hp.optim.scheduler.gamma)

    def compute_logits(self, feats, scope: str='all', **model_kwargs):
        if scope == 'train':
            # Otherwise unseen classes will leak through batch norm
            attrs_mask = self.train_seen_mask
        elif scope == 'seen':
            attrs_mask = self.seen_mask
        elif scope == 'all':
            attrs_mask = None
        elif scope == 'unseen':
            attrs_mask = self.unseen_mask

        return self.model(feats, attrs_mask=attrs_mask, **model_kwargs)

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
                ausuc = np.nan
        elif dataset == 'test':
            # GZSL metrics
            logits = self.run_inference(self.test_dataloader, scope='all')
            logits[:, self.seen_mask] *= 0.95
            preds = logits.argmax(dim=1).numpy()
            guessed = (preds == self.test_labels)
            # seen_acc = guessed[self.test_seen_idx].mean()
            # unseen_acc = guessed[self.test_unseen_idx].mean()
            seen_acc = np.mean([guessed[cls_idx].mean().item() for cls_idx in [self.class_indices_inside_test[c] for c in self.seen_classes]])
            unseen_acc = np.mean([guessed[cls_idx].mean().item() for cls_idx in [self.class_indices_inside_test[c] for c in self.unseen_classes]])
            harmonic = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

            # ZSL
            zsl_logits = prune_logits(logits, self.unseen_mask)
            zsl_preds = zsl_logits.argmax(dim=1).numpy()
            # zsl_acc = (zsl_preds == self.test_labels)[self.test_unseen_idx].mean()
            zsl_guessed = (zsl_preds == self.test_labels)
            zsl_acc = np.mean([zsl_guessed[cls_idx].mean().item() for cls_idx in [self.class_indices_inside_test[c] for c in self.unseen_classes]])

            # AUSUC
            if self.config.get('logging.compute_ausuc'):
                ausuc = compute_ausuc(logits, self.test_labels, self.seen_mask) * 0.01
            else:
                ausuc = np.nan
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
            f'GZSL-U: {scores[1]:.1f}. ' \
            f'GZSL-S: {scores[0]:.1f}. ' \
            f'GZSL-H: {scores[2]:.1f}. ' \
            f'ZSL: {scores[3]:.1f}. ' \
            f'AUSUC: {scores[4]:.1f}.')
