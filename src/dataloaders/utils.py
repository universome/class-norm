import os
from os import PathLike
from typing import List, Tuple, Any, Callable

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from firelab.utils.training_utils import get_module_device
from torch.utils.data import Dataset
import torchvision.transforms.functional as TVF

from src.models.classifier import ResnetEmbedder
from src.models.layers import ResNetConvEmbedder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
imagenet_normalization = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD )
imagenet_denormalization = lambda x: x * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]


def read_column(filename:PathLike, column_idx:int) -> List[str]:
    with open(filename) as f:
        column = [line.split(' ')[column_idx] for line in f.read().splitlines()]

    return column


def shuffle_dataset(imgs: List[Any], labels: List[int]) -> Tuple[List[Any], List[int]]:
    assert len(imgs) == len(labels)

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
    imgs = [img.transpose(2, 0, 1) for img in tqdm(imgs, desc='[Transposing]')]
    imgs = [imagenet_normalization(torch.tensor(img) / 255).numpy() for img in tqdm(imgs, desc='[Normalizing]')]

    return imgs


def extract_resnet_features_for_dataset(
    dataset: List[Tuple[np.ndarray, int]],
    resnet_n_layers: int=18,
    feat_level: str='fc',
    device: str='cpu',
    *args, **kwargs) -> List[Tuple[np.ndarray, int]]:

    if feat_level == 'fc':
        embedder = ResnetEmbedder(resnet_n_layers=resnet_n_layers, pretrained=True).to(device)
    elif feat_level == 'conv':
        embedder = ResNetConvEmbedder(resnet_n_layers=resnet_n_layers, pretrained=True).to(device)
    else:
        raise NotImplementedError(f'Unknown feat level: {feat_level}')

    return extract_features_for_dataset(dataset, embedder, *args, **kwargs)


def extract_features_for_dataset(
    dataset: List[Tuple[np.ndarray, int]],
    embedder: nn.Module,
    device: str='cpu',
    batch_size: int=64) -> List[Tuple[np.ndarray, int]]:

    embedder = embedder.eval()
    embedder = embedder.to(device)

    imgs = [x for x, _ in dataset]
    features = extract_features(imgs, embedder, batch_size=batch_size)

    return list(zip(features, [y for _, y in dataset]))


def extract_features(imgs: List[np.ndarray], embedder: nn.Module, batch_size: int=64) -> List[np.ndarray]:
    dataloader = DataLoader(imgs, batch_size=batch_size)
    device = get_module_device(embedder)
    result = []

    with torch.no_grad():
        for x in tqdm(dataloader, desc='[Extracting features]'):
            feats = embedder(x.to(device)).cpu().numpy()
            result.extend(feats)

    return result


class CustomDataset(Dataset):
    def __init__(self, dataset: List, transform: Callable=None):
        # self.dataset = load_dataset(data_dir, split=('train' if train else 'test'))
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = x.astype(np.uint8)

        if not self.transform is None:
            x = self.transform(x)

        return x, y

    def __len__(self) -> int:
        return len(self.dataset)


class CenterCropToMin(object):
    """
    CenterCrops an image to a min size
    """
    def __call__(self, img):
        assert TVF._is_pil_image(img)

        return TVF.center_crop(img, min(img.size))
