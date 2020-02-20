import os
import pickle
from os import PathLike
from typing import List, Tuple, Callable

import numpy as np
from torch.utils.data import Dataset

from .utils import read_column, shuffle_dataset, load_imgs, preprocess_imgs


def load_dataset(
        data_dir: PathLike,
        split: str='train',
        target_shape: Tuple[int, int]=None,
        preprocess: bool=True) -> List[Tuple[np.ndarray, int]]:

    filename = os.path.join(data_dir, 'images.txt')
    img_paths = read_column(filename, 1)
    train_test_split = load_train_test_split(data_dir)
    img_paths = [p for (p, train) in zip(img_paths, train_test_split) if split == 'train']

    # import random
    # img_paths = random.sample(img_paths, 100)

    imgs = load_imgs(os.path.join(data_dir, 'images'), img_paths, target_shape)
    labels = load_labels(img_paths)
    if preprocess: imgs = preprocess_imgs(imgs)
    if split == 'train': imgs, labels = shuffle_dataset(imgs, labels)

    return list(zip(imgs, labels))


def load_train_test_split(data_dir: PathLike) -> List[bool]:
    filepath = os.path.join(data_dir, 'train_test_split.txt')
    train_test_split = read_column(filepath, 1)
    train_test_split = [int(f) > 0 for f in train_test_split]

    return train_test_split


def load_labels(img_paths:List[PathLike]) -> List[int]:
    # Class index is encoded in the path. Let's use it.
    return [(int(p.split('.')[0]) - 1) for p in img_paths]


def load_class_attributes(data_dir: PathLike) -> np.ndarray:
    # filename = os.path.join(data_dir, 'attributes/class_attribute_labels_continuous.txt')
    #
    # with open(filename) as f:
    #     attrs = f.read().splitlines()
    #     attrs = [[float(a) for a in attr.split(' ')] for attr in attrs]
    #     attrs = np.array(attrs)
    #     #attrs = (attrs - attrs.mean(axis=0)) / attrs.max(axis=0)
    #     attrs = attrs / (attrs.max(axis=0) * 5)
    #     # attrs /= (attrs.std(axis=0) * 50)

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
