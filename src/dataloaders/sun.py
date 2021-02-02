import os
from os import PathLike
from typing import List, Tuple, Callable

import numpy as np
from torch.utils.data import Dataset

from src.dataloaders.utils import read_column, shuffle_dataset, create_default_transform
from src.dataloaders.dataset import ImageDataset
from src.utils.constants import DEBUG

def load_dataset(
        data_dir: PathLike,
        split: str='train',
        target_shape: Tuple[int, int]=None,
        in_memory: bool=False) -> List[Tuple[np.ndarray, int]]:

    idx = np.load(os.path.join(data_dir, f'{split}_idx.npy'))
    img_paths = np.load(os.path.join(data_dir, 'image_files.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))

    img_paths = img_paths[idx]
    labels = labels[idx]

    if DEBUG:
        labels_list = labels.tolist()
        debug_idx = [labels_list.index(c) for c in range(717)]
        img_paths = img_paths[debug_idx]
        labels = labels[debug_idx]

    img_paths = [os.path.join(data_dir, 'images', p) for p in img_paths]
    labels = labels.tolist()

    return ImageDataset(img_paths, labels, create_default_transform(target_shape), in_memory=in_memory)


def load_class_attributes(data_dir: os.PathLike) -> np.ndarray:
    return np.load(os.path.join(data_dir, 'attributes.npy'))
