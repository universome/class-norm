import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from firelab.base_trainer import BaseTrainer
from firelab.config import Config

from src.utils.training_utils import construct_optimizer, prune_logits, normalize
from src.utils.data_utils import construct_output_mask


class ZSLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        config = config.clone(frozen=False)
        config.data = config.data[config.dataset]
        config.exp_name = f'zsl_{config.dataset}'
        super().__init__(config)

    def init_dataloaders(self):
        feats = np.load(f'{self.config.data.dir}/feats.npy').astype(np.float32)
        labels = np.load(f'{self.config.data.dir}/labels.npy').astype(int)
        attrs = np.load(f'{self.config.data.dir}/attrs.npy').astype(np.float32)
        train_idx = np.load(f'{self.config.data.dir}/train_idx.npy')
        test_idx = np.load(f'{self.config.data.dir}/test_idx.npy')

        ds_train = [(feats[i], labels[i]) for i in train_idx]
        ds_test = [(feats[i], labels[i]) for i in test_idx]

        self.train_dataloader = DataLoader(ds_train, batch_size=self.config.hp.batch_size, shuffle=True, num_workers=0)
        self.val_dataloader = DataLoader(ds_test, batch_size=2048, num_workers=0)
        self.attrs = torch.from_numpy(attrs).to(self.device_name)
        # self.attrs = (self.attrs - self.attrs.mean(dim=0, keepdim=True)) / self.attrs.std(dim=0, keepdim=True)

        self.seen_mask = construct_output_mask(self.config.data.seen_classes, self.config.data.num_classes)
        self.unseen_mask = construct_output_mask(self.config.data.seen_classes, self.config.data.num_classes)

        self.test_labels = np.array([labels[i] for i in test_idx])
        self.test_seen_idx = [i for i, y in enumerate(self.test_labels) if y in self.config.data.seen_classes]
        self.test_unseen_idx = [i for i, y in enumerate(self.test_labels) if y in self.config.data.unseen_classes]

    def _run_training(self):
        from tqdm import tqdm

        for epoch in range(1, self.config.hp.max_num_epochs + 1):
            for batch in tqdm(self.train_dataloader):
                self.train_on_batch(batch)

            if epoch % self.config.val_freq_epochs == 0:
                self.validate()

    def train_on_batch(self, batch):
        self.model.train()
        feats = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        labels = torch.from_numpy(np.array(batch[1])).to(self.device_name)
        logits = self.compute_logits(feats, prune='seen')
        loss = F.cross_entropy(logits, labels)

        self.optim.zero_grad()
        loss.backward()
        if self.config.hp.get('grad_clip_val', 0) > 0:
            clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_val)
        self.optim.step()

    def compute_logits(self, feats, prune: str='none'):
        if self.config.hp.get('attrs_dropout.p') and self.model.training:
            mask = (torch.rand(self.attrs.shape[1]) > self.config.hp.attrs_dropout.p)
            mask = mask.unsqueeze(0).float().to(self.attrs.device)
            attrs = self.attrs * mask
            attrs = attrs / (1 - self.config.hp.attrs_dropout.p)
        else:
            attrs = self.attrs

        if self.config.hp.get('feats_dropout.p') and self.model.training:
            feats = F.dropout(feats, p=self.config.hp.feats_dropout.p)

        protos = self.model(attrs)
        feats = normalize(feats, scale_value=self.config.hp.scale)
        protos = normalize(protos, scale_value=self.config.hp.scale)
        logits = feats @ protos.t()

        if prune == 'seen':
            logits = prune_logits(logits, self.seen_mask)
        elif prune == 'unseen':
            logits = prune_logits(logits, self.unseen_mask)

        return logits

    def init_models(self):
        self.model = nn.Sequential(
            nn.Linear(self.attrs.shape[1], self.config.hp.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hp.hid_dim, self.config.hp.feat_dim),
            nn.ReLU()
        ).to(self.device_name)

        # self.model = nn.Sequential(
        #     nn.Linear(self.attrs.shape[1], self.config.hp.feat_dim),
        # ).to(self.device_name)

        self.init_output_layer(self.model[2])

    def init_output_layer(self, layer: nn.Linear):
        # attrs_mean_norm = self.attrs.norm(dim=1).pow(2).mean()
        std = 1 / np.sqrt(self.config.hp.hid_dim * self.config.hp.feat_dim)
        layer.weight.data.normal_(0, std)

    def init_optimizers(self):
        self.optim = construct_optimizer(self.model.parameters(), self.config.hp.optim)

    def run_inference(self, dataloader: DataLoader, prune: str='none'):
        with torch.no_grad():
            logits = [self.compute_logits(x.to(self.device_name), prune).cpu() for x, _ in dataloader]
        logits = torch.cat(logits, dim=0)

        return logits

    def validate(self):
        self.model.eval()
        logits = self.run_inference(self.val_dataloader)
        preds = logits.argmax(dim=1).numpy()
        guessed = (preds == self.test_labels)
        seen_acc = guessed[self.test_seen_idx].mean()
        unseen_acc = guessed[self.test_unseen_idx].mean()
        harmonic = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

        print(f'GZSL-S: {seen_acc: .4f}. GZSL-U: {unseen_acc: .4f}. GZSL-H: {harmonic: .4f}')
