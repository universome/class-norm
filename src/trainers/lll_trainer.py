from typing import List, Tuple
import random

import torch
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from tqdm import tqdm

from src.models.classifier import ZSClassifier, ResnetClassifier
from src.dataloaders import cub, awa
from src.utils.data_utils import split_classes_for_tasks, get_train_test_data_splits
from src.trainers.basic_task_trainer import BasicTaskTrainer
from src.trainers.agem_task_trainer import AgemTaskTrainer
from src.trainers.ewc_task_trainer import EWCTaskTrainer
from src.trainers.mas_task_trainer import MASTaskTrainer


class LLLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super(LLLTrainer, self).__init__(config)

        self.logger.info(f'Implementation method: {self.config.task_trainer}')
        self.logger.info(f'Using device: {self.device_name}')

        self.episodic_memory = []
        self.episodic_memory_output_mask = []
        self.zst_accs = []
        self.accs_history = []

    def init_models(self):
        if self.config.hp.get('use_class_attrs'):
            self.model = ZSClassifier(self.class_attributes, pretrained=self.config.hp.pretrained)
        else:
            self.model = ResnetClassifier(self.config.data.num_classes, pretrained=self.config.hp.pretrained)

        self.model = self.model.to(self.device_name)

    def init_optimizers(self):
        # TODO: without momentum?!
        self.optim = torch.optim.SGD(self.model.parameters(), **self.config.hp.optim_kwargs.to_dict())

    def init_dataloaders(self):
        if self.config.data.name == 'CUB':
            self.ds_train = cub.load_dataset(self.config.data.dir, is_train=True, target_shape=self.config.hp.img_target_shape)
            self.ds_test = cub.load_dataset(self.config.data.dir, is_train=False, target_shape=self.config.hp.img_target_shape)
            self.class_attributes = cub.load_class_attributes(self.config.data.dir)
        elif self.config.data.name == 'AWA':
            self.ds_train = awa.load_dataset(self.config.data.dir, split='train', target_shape=self.config.hp.img_target_shape)
            self.ds_test = awa.load_dataset(self.config.data.dir, split='test', target_shape=self.config.hp.img_target_shape)
            self.class_attributes = awa.load_class_attributes(self.config.data.dir)
        else:
            raise NotImplementedError(f'Unkown dataset: {self.config.data.name}')

        self.class_splits = split_classes_for_tasks(self.config.data.num_classes, self.config.hp.num_tasks)
        self.data_splits = get_train_test_data_splits(self.class_splits, self.ds_train, self.ds_test)

        for task_idx, split in enumerate(self.class_splits):
            print(f'[Task {task_idx}]:', self.class_splits[task_idx].tolist())

    def start(self):
        self.init()
        self.num_tasks_learnt = 0
        self.task_trainers = [] # TODO: this is memory-leaky :|

        for task_idx in range(self.config.hp.num_tasks):
            print(f'Starting task #{task_idx}', end='')

            task_trainer = self.construct_trainer(task_idx)

            self.task_trainers.append(task_trainer)
            self.zst_accs.append(task_trainer.compute_test_accuracy())
            print(f'. ZST accuracy: {self.zst_accs[-1]}')
            task_trainer.start()
            self.num_tasks_learnt += 1
            #self.validate()
            print(f'Train accuracy: {task_trainer.compute_train_accuracy()}')
            print(f'Test accuracy: {task_trainer.compute_test_accuracy()}')

            if self.config.task_trainer == 'agem':
                task_trainer.extend_episodic_memory()

        print('ZST accs:', self.zst_accs)

    def construct_trainer(self, task_idx: int) -> "TaskTrainer":
        if self.config.task_trainer == 'basic':
            return BasicTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'agem':
            return AgemTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'ewc':
            return EWCTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'mas':
            return MASTaskTrainer(self, task_idx)
        else:
            raise NotImplementedError(f'Unknown task trainer: {self.config.task_trainer}')

    def validate(self):
        """Computes model accuracy on all the tasks"""
        accs = []

        for task_idx in tqdm(range(self.config.hp.num_tasks), desc='[Validating]'):
            trainer = self.construct_trainer(task_idx)
            acc = trainer.compute_test_accuracy()
            accs.append(acc)
            self.writer.add_scalar('Task_test_acc/{}', acc, self.num_tasks_learnt)

        self.accs_history.append(accs)
        print('Accuracies:', accs)
