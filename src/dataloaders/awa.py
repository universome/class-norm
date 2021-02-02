import random
import os
import pickle
from os import PathLike
from typing import List, Tuple

import numpy as np

from src.utils.constants import DEBUG
from src.dataloaders.utils import read_column, create_default_transform
from src.dataloaders.dataset import ImageDataset


def load_dataset_paths(data_dir: PathLike, split: str) -> List[Tuple[os.PathLike, int]]:
    assert split in ['train', 'val', 'test'], f"Unknown dataset split: {split}"

    filename = os.path.join(data_dir, f'AWA_{split}_list.txt')
    img_paths = [os.path.join(data_dir, p) for p in read_column(filename, 0)]
    labels = read_column(filename, 1)
    labels = [int(l) for l in labels]

    if DEBUG:
        chosen_idx = random.sample(range(len(labels)), 200)
        img_paths = [img_paths[i] for i in chosen_idx]
        labels = [labels[i] for i in chosen_idx]

    return list(zip(img_paths, labels))


def load_dataset(data_dir: PathLike,
                 split: str,
                 target_shape: Tuple[int, int]=None) -> List[Tuple[np.ndarray, int]]:

    img_paths, labels = zip(*load_dataset_paths(data_dir, split))

    return ImageDataset(img_paths, labels, create_default_transform(target_shape))


def load_class_attributes(data_dir: PathLike) -> np.ndarray:
    filename = os.path.join(data_dir, 'AWA_attr_in_order.pickle')

    with open(filename, 'rb') as f:
        attrs = pickle.load(f, encoding='latin1')

    return attrs
