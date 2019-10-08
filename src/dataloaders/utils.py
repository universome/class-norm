import os
from os import PathLike
from typing import List, Tuple, Any

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms


imagenet_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def read_column(filename:PathLike, column_idx:int) -> List[str]:
    with open(filename) as f:
        column = [line.split(' ')[column_idx] for line in f.read().splitlines()]

    return column


def shuffle_dataset(imgs: List[Any], labels: List[int]) -> Tuple[List[Any], List[int]]:
    shuffling = np.random.permutation(len(imgs))
    imgs = [imgs[i] for i in shuffling]
    labels = [labels[i] for i in shuffling]

    return imgs, labels


def load_imgs(image_folder: PathLike, img_paths: List[PathLike], target_shape=None) -> List[np.ndarray]:
    full_img_paths = [os.path.join(image_folder, p) for p in img_paths]
    images = [load_img(p, target_shape) for p in tqdm(full_img_paths, desc='[Loading dataset]')]

    return images


def load_img(img_path: PathLike, target_shape: Tuple[int, int]=None):
    img = cv2.imread(img_path)

    if target_shape != None:
        img = cv2.resize(img, target_shape)

    return img.astype(np.float32)


def preprocess_imgs(imgs: List[np.ndarray]) -> List[np.ndarray]:
    imgs = [img.transpose(2, 0, 1) for img in tqdm(imgs, desc='[Reshaping]')]
    imgs = [imagenet_normalization(torch.tensor(img) / 255).numpy() for img in tqdm(imgs, desc='Normalizing')]

    return imgs
