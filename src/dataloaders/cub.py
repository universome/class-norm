import os
import pickle
from os import PathLike
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm
import torch
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_cub_dataset(cub_data_dir: PathLike, is_train:bool=True, target_shape=(224, 224)) -> List[Tuple[np.ndarray, int]]:
    filename = os.path.join(cub_data_dir, 'images.txt')
    img_paths = read_second_column(filename)
    train_test_split = load_train_test_split(cub_data_dir)
    img_paths = [p for (p, train) in zip(img_paths, train_test_split) if train == is_train]
    imgs = load_imgs(cub_data_dir, img_paths)
    labels = load_labels(img_paths)

    imgs = [cv2.resize(img, target_shape) for img in tqdm(imgs, desc='[Resizing]')]
    imgs = [img.transpose(2, 0, 1) for img in tqdm(imgs, desc='[Reshaping]')]
    imgs = [normalize(torch.tensor(img) / 255).numpy() for img in tqdm(imgs, desc='Normalizing')]

    shuffling = np.random.permutation(len(imgs))
    imgs = [imgs[i] for i in shuffling]
    labels = [labels[i] for i in shuffling]

    return list(zip(imgs, labels))


def load_train_test_split(cub_data_dir: PathLike) -> List[bool]:
    filepath = os.path.join(cub_data_dir, 'train_test_split.txt')
    train_test_split = read_second_column(filepath)
    train_test_split = [int(f) > 0 for f in train_test_split]

    return train_test_split


def read_second_column(filename:PathLike) -> List[str]:
    with open(filename) as f:
        second_column = [line.split(' ')[1] for line in f.read().splitlines()]

    return second_column


def load_imgs(cub_data_dir: PathLike, img_paths: List[PathLike]) -> List[np.ndarray]:
    full_img_paths = [os.path.join(cub_data_dir, 'images', p) for p in img_paths]
    images = [cv2.imread(p).astype(np.float32) for p in tqdm(full_img_paths, desc='[Loading CUB dataset]')]

    return images


def load_labels(img_paths:List[PathLike]) -> List[int]:
    # Class index is encoded in the path. Let's use it.
    return [(int(p.split('.')[0]) - 1) for p in img_paths]


def load_class_attributes(cub_data_dir: PathLike) -> np.ndarray:
    # filename = os.path.join(cub_data_dir, 'attributes/class_attribute_labels_continuous.txt')
    #
    # with open(filename) as f:
    #     attrs = f.read().splitlines()
    #     attrs = [[float(a) for a in attr.split(' ')] for attr in attrs]
    #     attrs = np.array(attrs)
    #     #attrs = (attrs - attrs.mean(axis=0)) / attrs.max(axis=0)
    #     attrs = attrs / (attrs.max(axis=0) * 5)
    #     # attrs /= (attrs.std(axis=0) * 50)

    filename = os.path.join(cub_data_dir, 'CUB_attr_in_order.pickle')

    with open(filename, 'rb') as f:
        attrs = pickle.load(f, encoding='latin1')

    return attrs
