from typing import List, Tuple
import random

import torch
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from tqdm import tqdm

from src.models.classifier import ZSClassifier
from src.dataloaders.cub import load_cub_dataset, load_class_attributes
from src.utils.data_utils import (
    split_classes_for_tasks,
    get_train_test_data_splits,
    construct_output_mask,
)
from src.trainers.agem_task_trainer import AgemTaskTrainer
from src.trainers.ewc_task_trainer import EWCTaskTrainer
# from src.trainers.mas_task_trainer import MASTaskTrainer


class LLLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super(LLLTrainer, self).__init__(config)

        self.episodic_memory = []
        self.episodic_memory_output_mask = []
        self.zst_accs = []
        self.accs_history = []

    def init_models(self):
        self.model = ZSClassifier(load_class_attributes(self.config.data_dir), pretrained=self.config.hp.pretrained)
        self.model = self.model.to(self.device_name)

    def init_optimizers(self):
        # TODO: without momentum?!
        self.optim = torch.optim.SGD(self.model.parameters(), **self.config.hp.optim_kwargs.to_dict())

    def init_dataloaders(self):
        self.ds_train = load_cub_dataset(self.config.data_dir, is_train=True, target_shape=self.config.hp.img_target_shape)
        self.ds_test = load_cub_dataset(self.config.data_dir, is_train=False, target_shape=self.config.hp.img_target_shape)
        self.class_splits = split_classes_for_tasks(self.config.num_classes, self.config.hp.num_tasks)
        self.data_splits = get_train_test_data_splits(self.class_splits, self.ds_train, self.ds_test)

        for task_idx, split in enumerate(self.class_splits):
            print(f'[Task {task_idx}]:', self.class_splits[task_idx].tolist())

    def start(self):
        self.init()
        self.num_tasks_learnt = 0

        for task_idx in range(self.config.hp.num_tasks):
            print(f'Starting task #{task_idx}', end='')

            if self.config.task_trainer == 'agem':
                task_trainer = AgemTaskTrainer(self, task_idx)
            elif self.config.task_trainer == 'ewc':
                task_trainer = EWCTaskTrainer(self, task_idx)
            else:
                raise NotImplementedError(f'Unknown task trainer: {self.config.task_trainer}')

            self.zst_accs.append(task_trainer.compute_test_accuracy())
            print(f'. ZST accuracy: {self.zst_accs[-1]}')
            task_trainer.start()
            self.num_tasks_learnt += 1
            #self.validate()
            print(f'Train accuracy: {task_trainer.compute_train_accuracy()}')
            print(f'Test accuracy: {task_trainer.compute_test_accuracy()}')

            self.extend_episodic_memory(task_idx, self.config.hp.num_mem_samples_per_class)

        print('ZST accs:', self.zst_accs)

    def validate(self):
        """Computes model accuracy on all the tasks"""
        accs = []

        for task_idx in tqdm(range(self.config.hp.num_tasks), desc='[Validating]'):
            trainer = AgemTaskTrainer(self, task_idx)
            acc = trainer.compute_test_accuracy()
            accs.append(acc)
            self.writer.add_scalar('Task_test_acc/{}', acc, self.num_tasks_learnt)

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