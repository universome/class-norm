import os
from typing import Tuple, List

import numpy as np
from torchvision.datasets import LSUN
import cv2
from tqdm import tqdm

from .utils import shuffle_dataset


def load_dataset(data_dir: os.PathLike, classes: List[str], target_shape: Tuple[int, int]=None):
    ds = LSUN(data_dir, classes=classes)
    # imgs, targets = zip(*[ds[i] for i in tqdm(range(ds.length), desc='[Loading]')]) # Loading into memory :|
    import random
    imgs, targets = zip(*[ds[i] for i in tqdm(random.sample(range(ds.length), k=1000), desc='[Loading]')]) # Loading into memory :|
    imgs = [(x / 127.5 - 1).astype(np.float32) for x in imgs]

    if not target_shape is None:
        imgs = [cv2.resize(img, target_shape) for img in tqdm(imgs, desc='[Resizing]')]

    return list(zip(*shuffle_dataset(imgs, targets)))
