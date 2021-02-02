import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.data_utils import construct_output_mask, flatten, remap_targets
from src.utils.training_utils import prune_logits
from src.trainers.task_trainer import TaskTrainer


class JointTaskTrainer(TaskTrainer):
    """
    Perfect score one can achieve: train on all the previous data
    """
    def _after_init_hook(self):
        seen_classes = np.unique(flatten(self.main_trainer.class_splits[:self.task_idx + 1]))

        self.task_ds_train = [ds_train for ds_train, ds_test in self.main_trainer.data_splits[:self.task_idx+1]]
        self.task_ds_train = [(x, y) for ds in self.task_ds_train for (x, y) in ds]

        self.joint_output_mask = construct_output_mask(seen_classes, self.config.lll_setup.num_classes)
        self.original_train_dataloader = self.train_dataloader
        self.train_dataloader = DataLoader(self.task_ds_train, batch_size=self.config.hp.batch_size,
                                           collate_fn=lambda b: list(zip(*b)), shuffle=True)

    def train_on_batch(self, batch):
        self.model.train()
        loss = self.compute_loss(self.model, batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def compute_train_accuracy(self):
        return self.compute_accuracy(self.original_train_dataloader)
