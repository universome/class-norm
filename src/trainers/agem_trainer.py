from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
import numpy as np

from src.models.classifier import ZSClassifier
from src.dataloaders.cub import load_cub_dataset, load_class_attributes
from src.utils.data_utils import (
    split_classes_for_tasks,
    get_train_test_data_splits,
    construct_output_mask,
    resize_dataset,
)


NEG_INF = float('-inf')


class AgemTrainer(BaseTrainer):
    def __init__(self, config):
        super(AgemTrainer, self).__init__(config)

        self.episodic_memory = []
        self.episodic_memory_output_mask = []
        self.zst_accs = []
        self.accs_history = []

    def init_models(self):
        self.model = ZSClassifier(load_class_attributes(self.config.data_dir)[:30], pretrained=self.config.hp.pretrained)
        self.model = self.model.to(self.device_name)

    def init_optimizers(self):
        # TODO: without momentum?!
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.03)

    def init_dataloaders(self):
        self.ds_train = load_cub_dataset(self.config.data_dir, is_train=True)
        self.ds_test = load_cub_dataset(self.config.data_dir, is_train=False)
        self.ds_train = resize_dataset(self.ds_train, self.config.hp.img_width, self.config.hp.img_height)
        self.ds_test = resize_dataset(self.ds_test, self.config.hp.img_width, self.config.hp.img_height)
        self.ds_train = [(x.transpose(2, 0, 1).astype(np.float32), y) for x, y in self.ds_train]
        self.ds_test = [(x.transpose(2, 0, 1).astype(np.float32), y) for x, y in self.ds_test]
        self.class_splits = split_classes_for_tasks(self.config.num_classes, self.config.hp.num_tasks)
        self.data_splits = get_train_test_data_splits(self.class_splits, self.ds_train, self.ds_test)

    def start(self):
        self.init()

        for task_idx in range(self.config.hp.num_tasks):
            trainer = AgemTaskTrainer(self, task_idx)
            self.zst_accs.append(trainer.compute_test_accuracy())
            trainer.start()
            self.validate()

            self.extend_episodic_memory(task_idx, self.config.hp.num_mem_samples_per_class)

    def validate(self):
        """Computes model accuracy on all the tasks"""
        accs = []

        for task_idx in range(self.config.hp.num_tasks):
            trainer = AgemTaskTrainer(self, task_idx)
            acc = trainer.compute_test_accuracy()
            accs.append(acc)
            self.writer.add_scalar('Task_test_acc/{}', acc)

        self.accs_history.append(accs)
        print('Accuracies:', accs)

    def extend_episodic_memory(self, task_idx:int, num_samples_per_class:int):
        """
        Adds examples from the given task to episodic memory

        :param:
            - task_idx — task index
            - num_samples_per_class — max number of samples of each class to add
        """
        ds_train, _ = self.data_splits[task_idx]
        unique_labels = set([y for _,y in ds_train])
        groups = [[(x,y) for (x,y) in ds_train if y == label] for label in unique_labels] # Slow but concise
        num_samples_to_add = [min(len(g), num_samples_per_class) for g in groups]
        task_memory = [random.sample(g, n) for g, n in zip(groups, num_samples_to_add)]
        task_memory = [s for group in task_memory for s in group] # Flattening
        task_mask = construct_output_mask(self.class_splits[task_idx], self.config.num_classes)
        task_mask = task_mask.reshape(1, -1).repeat(len(task_memory), axis=0)

        assert len(task_memory) <= num_samples_per_class * len(groups)
        assert len(task_mask) <= num_samples_per_class * len(groups)

        self.episodic_memory.extend(task_memory)
        self.episodic_memory_output_mask.extend([m for m in task_mask])


class AgemTaskTrainer(BaseTrainer):
    def __init__(self, main_trainer:AgemTrainer, task_idx:int):
        super(AgemTaskTrainer, self).__init__(main_trainer.config)

        self.task_idx = task_idx
        self.config = main_trainer.config
        self.model = main_trainer.model
        self.optim = main_trainer.optim
        self.episodic_memory = main_trainer.episodic_memory
        self.episodic_memory_output_mask = main_trainer.episodic_memory_output_mask
        self.criterion = nn.CrossEntropyLoss()

        self.task_ds_train, self.task_ds_test = main_trainer.data_splits[task_idx]
        self.output_mask = construct_output_mask(main_trainer.class_splits[task_idx], self.config.num_classes)
        self.init_dataloaders()

    def init_dataloaders(self):
        self.train_dataloader = DataLoader(self.task_ds_train, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)))
        self.test_dataloader = DataLoader(self.task_ds_test, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)))

    def train_on_batch(self, batch:Tuple[Tensor, Tensor]):
        x = torch.tensor(batch[0]).to(self.device_name)
        y = torch.tensor(batch[1]).to(self.device_name)

        logits = self.model(x)
        pruned_logits = self.prune_logits(logits)
        loss = self.criterion(pruned_logits, y)

        if len(self.episodic_memory) > 0:
            ref_grad = self.compute_ref_grad()
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()])
            grad = self.project_grad(grad, ref_grad)

            self.optim.zero_grad()
            self.set_grad(grad)
        else:
            self.optim.zero_grad()
            loss.backward()

        self.optim.step()

    def compute_ref_grad(self):
        num_samples_to_use = min(self.config.hp.mem_batch_size, len(self.episodic_memory))
        batch_idx = random.sample(np.arange(len(self.episodic_memory)).tolist(), num_samples_to_use)
        batch = [m for i, m in enumerate(self.episodic_memory) if i in batch_idx]
        output_mask = np.array([m for i, m in enumerate(self.episodic_memory_output_mask) if i in batch_idx])

        x = torch.tensor([x for x,_ in batch]).to(self.device_name)
        y = torch.tensor([y for _, y in batch]).to(self.device_name)
        logits = self.model(x)
        pruned_logits = logits.masked_fill(torch.tensor(~output_mask).to(self.device_name), NEG_INF)
        loss = self.criterion(pruned_logits, y)
        loss.backward()
        ref_grad = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()])
        self.optim.zero_grad()

        return ref_grad

    def project_grad(self, grad, ref_grad):
        dot_product = torch.dot(grad, ref_grad)

        if dot_product > 0:
            return grad
        else:
            return grad - ref_grad * dot_product / ref_grad.pow(2).sum()

    def set_grad(self, grad: Tensor):
        """Takes gradient and sets it to parameters .grad"""
        assert grad.dim() == 1

        for param in self.model.parameters():
            param.grad.data = grad[:param.numel()].view(*param.shape).data
            grad = grad[param.numel():]

        assert len(grad) == 0, "Not all weights were used!"

    def prune_logits(self, logits:Tensor) -> Tensor:
        """
        Takes logits and sets those classes which do not participate
        in the current task to -infinity so they are not explicitly penalized and forgotten
        """
        mask_idx = np.nonzero(~self.output_mask)[0]
        pruned = logits.index_fill(1, torch.tensor(mask_idx).to(self.device_name), NEG_INF)

        return pruned

    def compute_test_accuracy(self):
        guessed = []

        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = torch.tensor(x).to(self.device_name)
                y = torch.tensor(y).to(self.device_name)

                logits = self.model(x)
                pruned_logits = self.prune_logits(logits)

                guessed.extend((pruned_logits.argmax(dim=1) == y).cpu().data.tolist())

        return np.mean(guessed)
