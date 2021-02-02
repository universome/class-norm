import os
from typing import Tuple

import numpy as np
from torchvision.datasets import MNIST

from src.dataloaders.utils import shuffle_dataset

def load_dataset(data_dir: os.PathLike, split: str):
    ds = MNIST(data_dir, train=(split == 'train'))
    imgs = [(x.numpy() / 127.5 - 1).astype(np.float32).reshape(-1) for x in ds.data]
    ds = list(zip(*shuffle_dataset(imgs, ds.targets)))

    return ds
