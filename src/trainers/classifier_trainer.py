import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet18, resnet50
import torchvision.transforms as T
from firelab.base_trainer import BaseTrainer
from firelab.config import Config

from src.dataloaders import cub, awa
from src.dataloaders.load_data import load_data
from src.dataloaders.utils import CustomDataset, CenterCropToMin
from src.utils.losses import LabelSmoothingLoss
from src.models.classifier import resnet_embedder_forward
from src.utils.model_utils import filter_params
from src.utils.constants import INPUT_DIMS


RESNETS = {18: resnet18, 50: resnet50}


class ClassifierTrainer(BaseTrainer):
    """
    Just a normal classifier trainer
    """
    def __init__(self, config: Config):
        super(ClassifierTrainer, self).__init__(config)

    def init_models(self):
        if self.config.hp.model.type == 'resnet':
            # self.model = nn.Sequential(
            #     ResnetEmbedder(
            #         self.config.hp.model.n_resnet_layers,
            #         self.config.hp.model.pretrained),
            #     nn.Linear(INPUT_DIMS[f'resnet{self.config.hp.model.n_resnet_layers}_feat'], self.config.data.num_classes)
            # )
            self.model = RESNETS[self.config.hp.model.n_resnet_layers](pretrained=self.config.hp.model.pretrained)
            self.model.fc = nn.Linear(self.model.fc.weight.shape[1], self.config.data.num_classes)
            nn.init.kaiming_normal_(self.model.fc.weight.data)
        elif self.config.hp.model.type == 'resnet-head':
            self.model = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(INPUT_DIMS[self.config.hp.model.input_type], self.config.hp.model.hid_dim),
                nn.ReLU(),
                nn.Linear(self.config.hp.model.hid_dim, self.config.data.num_classes)
            )
        else:
            raise NotImplementedError(f'Unknown model: {self.config.hp.model.type}')

        self.model = self.model.to(self.device_name)

    def init_dataloaders(self):
        img_train_transform = T.Compose([
            T.ToPILImage(),
            #T.RandomResizedCrop(self.config.hp.img_target_shape, scale=(0.2, 1.0)),
            CenterCropToMin(),
            T.RandomHorizontalFlip(),
            T.Resize(self.config.hp.img_target_shape),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        img_test_transform = T.Compose([
            T.ToPILImage(),
            CenterCropToMin(),
            T.Resize(self.config.hp.img_target_shape),
            #T.CenterCrop(self.config.hp.img_target_shape),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        if self.config.data.name == 'CUB':
            ds_train = cub.load_dataset(self.config.data.dir, split='train')
            ds_test = cub.load_dataset(self.config.data.dir, split='test')
            ds_train = CustomDataset(ds_train, img_train_transform)
            ds_test = CustomDataset(ds_test, img_test_transform)
        elif self.config.data.name == 'AWA':
            ds_train = awa.load_dataset(self.config.data.dir, split='train')
            ds_test = awa.load_dataset(self.config.data.dir, split='test')
            ds_train = CustomDataset(ds_train, img_train_transform)
            ds_test = CustomDataset(ds_test, img_test_transform)
        elif self.config.data.name == 'CUB_EMBEDDINGS':
            ds_train, ds_test, _ = load_data(self.config.data)
            ds_train = [(torch.from_numpy(x), y) for x, y in ds_train]
            ds_test = [(torch.from_numpy(x), y) for x, y in ds_test]
        else:
            raise NotImplementedError('Unknwon')

        self.train_dataloader = DataLoader(ds_train, batch_size=self.config.hp.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(ds_test, batch_size=self.config.hp.batch_size, shuffle=False)

    def init_optimizers(self):
        if self.config.hp.optim.type == 'adam':
            OptimClass = torch.optim.Adam
        elif self.config.hp.optim.type == 'rmsprop':
            OptimClass = torch.optim.RMSprop
        elif self.config.hp.optim.type == 'sgd':
            OptimClass = torch.optim.SGD
        else:
            raise NotImplementedError(f'Unknown optimizer: {self.config.hp.optim.type}')

        if self.config.hp.model.type == 'resnet':
            self.optim = OptimClass([
                {'params': filter_params(self.model, 'fc'), 'lr': 0.0005},
                {'params': self.model.fc.parameters(), 'lr': 0.005},
            ], **self.config.hp.optim.kwargs.to_dict())
        else:
            self.optim = OptimClass(self.model.parameters(), **self.config.hp.optim.kwargs.to_dict())

        if self.config.hp.optim.has('scheduler'):
            assert self.config.hp.optim.scheduler.type == "step"

            self.has_scheduler = True
            self.scheduler = StepLR(optimizer=self.optim, **self.config.hp.optim.scheduler.kwargs.to_dict())
        else:
            self.has_scheduler = False

    def init_criterions(self):
        # self.criterion = LabelSmoothingLoss(self.config.lll_setup.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def on_epoch_done(self):
        if self.has_scheduler:
            self.scheduler.step()

    def train_on_batch(self, batch):
        self.model.train()

        x = batch[0].to(self.device_name)
        y = batch[1].to(self.device_name)

        # with torch.no_grad():
        #     feats = resnet_embedder_forward(self.model, x)
        # logits = self.model.fc(feats)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.num_iters_done)
        self.writer.add_scalar('train/acc', acc.detach().cpu().item(), self.num_iters_done)
        self.writer.add_scalar('train/lr', self.optim.param_groups[0]['lr'], self.num_iters_done)

    def validate(self):
        self.model.eval()

        guessed = []
        losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                x = batch[0].to(self.device_name)
                y = batch[1].to(self.device_name)
                logits = self.model(x)

                losses.extend(F.cross_entropy(logits, y, reduction='none').cpu().tolist())
                guessed.extend((logits.argmax(dim=1) == y).cpu().tolist())

        self.writer.add_scalar('val/loss', np.mean(losses), self.num_iters_done)
        self.writer.add_scalar('val/acc', np.mean(guessed), self.num_iters_done)
