import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from firelab.base_trainer import BaseTrainer
from torchvision.datasets import SVHN

from src.dataloaders import svhn
from src.models.classifier import ResnetClassifier
from src.models.gan import GAN
from src.dataloaders.utils import shuffle_dataset
from src.dataloaders.gan_dataloader import GANDataloader


class GANClassifierTrainer(BaseTrainer):
    """
    Trains a classifier on top of outputs of a GAN model
    But we test it on real data.
    """
    def __init__(self, config):
        super(GANClassifierTrainer, self).__init__(config)

    def init_models(self):
        assert self.config.data.get('source') == 'gan_model'

        self.classifier = ResnetClassifier(num_classes=self.config.lll_setup.num_classes)
        self.gan = GAN(self.config.hp.gan_config)
        self.gan.load_state_dict(torch.load(self.config.hp.gan_checkpoint_path))

        self.classifier.to(self.device_name)
        self.gan.to(self.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss()

    def init_optimizers(self):
        self.optim = torch.optim.Adam(self.classifier.parameters())

    def init_dataloaders(self):
        if self.config.data.name == 'SVHN':
            # Train dataloader
            assert self.config.data.get('source') == 'gan_model'
            self.train_dataloader = self.create_gan_dataloader()

            # Test dataloader
            ds_test = svhn.load_dataset(self.config.data.dir, 'test')
            self.test_dataloader = DataLoader(ds_test, batch_size=self.config.hp.batch_size, shuffle=False)
        else:
            raise NotImplementedError(f'Unknown dataset: {self.config.data.name}')

    def train_on_batch(self, batch):
        x = batch[0].to(self.device_name)
        y = batch[1].to(self.device_name)

        logits = self.classifier(x)
        loss = self.criterion(logits, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def compute_test_accuracy(self):
        guessed = []

        for x, y in self.test_dataloader:
            logits = self.classifier(x)
            guessed.extend((logits.argmax(dim=1) == y).long().cpu().tolist())

        return np.mean(guessed)

    def validate(self):
        test_acc = self.compute_test_accuracy()
        self.writer.add_scalar('Test_acc', test_acc, self.num_iters_done)

    def create_gan_dataloader(self):
        def sample_fn():
            y = torch.randint(low=0, high=self.config.lll_setup.num_classes, size=(self.config.hp.batch_size,))
            y = y.to(self.device_name)

            return (self.gan.generator.sample(y), y)

        return GANDataloader(sample_fn, self.config.max_num_iters)
