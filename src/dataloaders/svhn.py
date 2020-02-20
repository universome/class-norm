import os
from typing import Tuple

import numpy as np
from torchvision.datasets import SVHN

from .utils import shuffle_dataset

def load_dataset(data_dir: os.PathLike, split: str):
    ds = SVHN(data_dir, split=split)
    imgs = [(x / 127.5 - 1).astype(np.float32) for x in ds.data]
    ds = list(zip(*shuffle_dataset(imgs, ds.labels)))

    return ds
