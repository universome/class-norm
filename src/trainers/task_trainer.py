import os
import random
from typing import List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from firelab.config import Config

from src.utils.data_utils import construct_output_mask, flatten
from src.utils.training_utils import construct_optimizer


class TaskTrainer:
    def __init__(self, main_trainer: "LLLTrainer", task_idx:int):
        self.task_idx = task_idx
        self.main_trainer = main_trainer
        self.device_name = main_trainer.device_name
        self.config = main_trainer.config

        self.init_models()
        self.init_writer()

        self.criterion = nn.CrossEntropyLoss()
        self.optim = self.construct_optimizer()
        self.attrs = self.model.attrs if hasattr(self.model, 'attrs') else None
        self.task_ds_train, self.task_ds_test = main_trainer.data_splits[task_idx]
        self.output_mask = construct_output_mask(main_trainer.class_splits[task_idx], self.config.lll_setup.num_classes)
        self.classes = self.main_trainer.class_splits[self.task_idx]
        self.learned_classes = np.unique(flatten(self.main_trainer.class_splits[:self.task_idx])).tolist()
        self.learned_classes_mask = construct_output_mask(self.learned_classes, self.config.data.num_classes)
        self.seen_classes = np.unique(flatten(self.main_trainer.class_splits[:self.task_idx + 1])).tolist()
        self.seen_classes_mask = construct_output_mask(self.seen_classes, self.config.data.num_classes)
        self.init_dataloaders()
        self.init_episodic_memory()
        self.test_acc_batch_history = []
        self.after_iter_done_callbacks = []
        self.num_iters_done = 0
        self.num_epochs_done = 0

        self._after_init_hook()

    def init_models(self):
        self.model = self.main_trainer.model

        if self.config.hp.get('reinit_after_each_task'):
            self.model.load_state_dict(self.main_trainer.create_model().state_dict())

    def init_writer(self):
        self.writer = SummaryWriter(os.path.join(self.main_trainer.paths.logs_path, f'task_{self.task_idx}'), flush_secs=5)

    def construct_optimizer(self):
        return construct_optimizer(self.model.parameters(), self.config.hp.optim)

    def init_dataloaders(self):
        self.train_dataloader = self.create_dataloader(self.task_ds_train, shuffle=True)
        self.test_dataloader = self.create_dataloader(self.task_ds_test, shuffle=False)

    def create_dataloader(self, dataset: List[Tuple[Any, int]], shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.config.hp.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: list(zip(*b)))

    def _after_init_hook(self):
        pass

    def _before_train_hook(self):
        pass

    def _after_train_hook(self):
        pass

    def on_epoch_done(self):
        pass

    def run_after_iter_done_callbacks(self):
        for callback in self.after_iter_done_callbacks:
            callback(self)

    def get_previous_trainer(self) -> "TaskTrainer":
        if self.task_idx == 0 or (self.task_idx - 1) >= len(self.main_trainer.task_trainers):
            return None
        else:
            return self.main_trainer.task_trainers[self.task_idx - 1]

    def init_episodic_memory(self):
        if self.get_previous_trainer() is None:
            self.episodic_memory = []
        else:
            self.episodic_memory = self.get_previous_trainer().episodic_memory

    def update_episodic_memory(self):
        pass

    @property
    def is_trainable(self) -> bool:
        return True

    def start(self):
        """Runs training"""
        self._before_train_hook()

        assert self.is_trainable, "We do not have enough conditions to train this Task Trainer"\
                                  "(for example, previous trainers was not finished or this trainer was already run)"

        if self.task_idx == 0:
            num_epochs = self.config.hp.get('base_task_max_num_epochs', self.config.hp.max_num_epochs)
        else:
            num_epochs = self.config.hp.max_num_epochs

        epochs = range(1, num_epochs + 1)
        if self._should_tqdm_epochs(): epochs = tqdm(epochs, desc=f'Task #{self.task_idx}')

        for epoch in epochs:
            batches = self.train_dataloader

            if not self._should_tqdm_epochs():
                batches = tqdm(batches, desc=f'Task #{self.task_idx} [epoch {epoch}/{num_epochs}]')

            for batch in batches:
                self.train_on_batch(batch)
                self.num_iters_done += 1
                self.run_after_iter_done_callbacks()

            self.num_epochs_done += 1
            self.on_epoch_done()

        self.update_episodic_memory()
        self._after_train_hook()

    def train_on_batch(self, batch):
        raise NotImplementedError

    def compute_accuracy(self, dataloader: DataLoader):
        guessed = []
        self.model.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x = torch.from_numpy(np.array(x)).to(self.device_name)
                y = torch.from_numpy(np.array(y)).to(self.device_name)

                pruned_logits = self.model.compute_pruned_predictions(x, self.output_mask)

                guessed.extend((pruned_logits.argmax(dim=1) == y).cpu().data.tolist())

        return np.mean(guessed)

    def compute_test_accuracy(self):
        return self.compute_accuracy(self.test_dataloader)

    def compute_train_accuracy(self):
        return self.compute_accuracy(self.train_dataloader)

    def _should_tqdm_epochs(self) -> bool:
        return self.config.hp.max_num_epochs > 10

    def sample_from_memory(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        samples = random.choices(self.episodic_memory, k=batch_size)
        x, y = zip(*samples)
        x = torch.from_numpy(np.array(x)).to(self.device_name)
        y = torch.tensor(y).to(self.device_name)

        return x, y
