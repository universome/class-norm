import os
from os import PathLike
from typing import List, Any

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from skimage.io import imread
from tqdm import tqdm


def load_cub_dataset(cub_data_dir: PathLike, is_train:bool=True) -> Dataset:
    filename = os.path.join(cub_data_dir, 'images.txt')
    img_paths = read_second_column(filename)
    train_test_split = load_train_test_split(cub_data_dir)
    img_paths = [p for (p, train) in zip(img_paths, train_test_split) if train == is_train]
    imgs = load_imgs(cub_data_dir, img_paths)
    labels = load_labels(img_paths)

    return list(zip(imgs, labels))


def load_train_test_split(cub_data_dir: PathLike) -> List[bool]:
    filepath = os.path.join(cub_data_dir, 'train_test_split.txt')
    train_test_split = read_second_column(filepath)
    train_test_split = [int(f) > 0 for f in train_test_split]

    return train_test_split


def read_second_column(filename:PathLike) -> List[Any]:
    with open(filename) as f:
        second_column = [line.split(' ')[1] for line in f.read().splitlines()]

    return second_column


def load_imgs(cub_data_dir: PathLike, img_paths: List[PathLike]) -> List[np.ndarray]:
    full_img_paths = [os.path.join(cub_data_dir, 'images', p) for p in img_paths]
    images = [imread(p) for p in tqdm(full_img_paths, desc='[Loading CUB dataset]')]

    return images


def load_labels(img_paths:List[PathLike]) -> List[int]:
    # Class index is encoded in the path. Let's use it.
    return [(int(p.split('.')[0]) - 1) for p in img_paths]


def load_class_attributes(cub_data_dir: PathLike) -> np.ndarray:
    filename = os.path.join(cub_data_dir, 'attributes/class_attribute_labels_continuous.txt')

    with open(filename) as f:
        attrs = f.read().splitlines()
        attrs = [[float(a) for a in attr.split(' ')] for attr in attrs]
        attrs = np.array(attrs)

    return attrs
