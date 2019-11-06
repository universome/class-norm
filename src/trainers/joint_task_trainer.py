import numpy as np
from torch.utils.data import DataLoader

from src.utils.data_utils import construct_output_mask

from .basic_task_trainer import BasicTaskTrainer


class JointTaskTrainer(BasicTaskTrainer):
    """
    Perfect score one can achieve: train on all the previous data
    """
    def _after_init_hook(self):
        seen_classes = np.unique(self.main_trainer.class_splits[:self.task_idx + 1])

        self.task_ds_train = [ds_train for ds_train, ds_test in self.main_trainer.data_splits[:self.task_idx+ 1]]
        self.task_ds_train = [(x, y) for ds in self.task_ds_train for (x, y) in ds]

        self.output_mask = construct_output_mask(seen_classes, self.config.data.num_classes)
        self.train_dataloader = DataLoader(self.task_ds_train, batch_size=self.config.hp.batch_size, collate_fn=lambda b: list(zip(*b)))
