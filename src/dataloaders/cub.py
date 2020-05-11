import os
import pickle
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

    filename = os.path.join(data_dir, 'images.txt')
    img_paths = read_column(filename, 1)
    train_test_split = load_train_test_split(data_dir)
    img_paths = [p for (p, img_split) in zip(img_paths, train_test_split) if split == img_split]
    labels = load_labels(img_paths)

    if DEBUG:
        # import random
        # img_paths = random.sample(img_paths, 200)
        debug_idx = [labels.index(c) for c in range(200)]
        img_paths = [img_paths[i] for i in debug_idx]
        labels = load_labels(img_paths)

    img_paths = [os.path.join(data_dir, 'images', p) for p in img_paths]

    return ImageDataset(img_paths, labels, create_default_transform(target_shape), in_memory=in_memory)


def load_preprocessed_dataset(data_dir: PathLike, split: str='train', **kwargs)-> List[Tuple[np.ndarray, int]]:
    if DEBUG:
        return load_dataset(data_dir, split, **kwargs)

    imgs = np.load(os.path.join(data_dir, f'{split}_images'))
    labels = np.load(os.path.join(data_dir, f'{split}_labels'))

    if split == 'train': imgs, labels = shuffle_dataset(imgs, labels)

    return list(zip(imgs, labels))


def load_train_test_split(data_dir: PathLike) -> List[str]:
    filepath = os.path.join(data_dir, 'train_test_split.txt')
    train_test_split = read_column(filepath, 1)
    train_test_split = [('train' if int(f) > 0 else 'test') for f in train_test_split]

    return train_test_split


def load_labels(img_paths:List[PathLike]) -> List[int]:
    # Class index is encoded in the path. Let's use it.
    return [(int(p.split('.')[0]) - 1) for p in img_paths]


def load_class_attributes(data_dir: PathLike, normalized: bool=False) -> np.ndarray:
    # filename = os.path.join(data_dir, 'attributes/class_attribute_labels_continuous.txt')
    #
    # with open(filename) as f:
    #     attrs = f.read().splitlines()
    #     attrs = [[float(a) for a in attr.split(' ')] for attr in attrs]
    #     attrs = np.array(attrs)
    #     #attrs = (attrs - attrs.mean(axis=0)) / attrs.max(axis=0)
    #     attrs = attrs / (attrs.max(axis=0) * 5)
    #     # attrs /= (attrs.std(axis=0) * 50)

    if normalized:
        return np.load(os.path.join(data_dir, 'attributes_normalized.npy'))
    else:
        filename = os.path.join(data_dir, 'CUB_attr_in_order.pickle')

        with open(filename, 'rb') as f:
            attrs = pickle.load(f, encoding='latin1')

        return attrs



class CUB(Dataset):
    def __init__(self, data_dir: str, train: bool=True, transform: Callable=None):
        self.dataset = load_dataset(data_dir, split=('train' if train else 'test'), preprocess=False)
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = x.astype(np.uint8)

        if not self.transform is None:
            x = self.transform(x)

        return x, y

    def __len__(self) -> int:
        return len(self.dataset)
