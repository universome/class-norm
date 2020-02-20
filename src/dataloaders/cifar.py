import os
from typing import Tuple

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

from src.dataloaders.utils import shuffle_dataset

dataset_classes = {10: CIFAR10, 100: CIFAR100}

def load_dataset(data_dir: os.PathLike, split: str, num_classes: int=10):
    ds = dataset_classes[num_classes](data_dir, train=(split == 'train'))
    imgs = [(x / 127.5 - 1).astype(np.float32).transpose(2, 0, 1) for x in ds.data]
    ds = list(zip(*shuffle_dataset(imgs, ds.targets)))

    return ds
