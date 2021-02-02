from typing import Tuple

import torch
from torch import Tensor

from src.utils.weights_importance import compute_mse_grad
from src.trainers.ewc_online_task_trainer import EWCOnlineTaskTrainer


class MASTaskTrainer(EWCOnlineTaskTrainer):
    def compute_importances(self, dataloader, output_mask):
        return compute_mse_grad(self.model, dataloader, output_mask)
