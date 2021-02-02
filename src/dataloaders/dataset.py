from os import PathLike
from typing import List, Callable, Tuple, Iterable

import numpy as np
from torch.utils.data import Dataset

from src.dataloaders.utils import load_img, load_imgs


class ImageDataset(Dataset):
    def __init__(self, img_paths: List[PathLike], labels: List[int], transform: Callable=None, in_memory: bool=False):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.in_memory = in_memory

        if self.in_memory:
            self.cached_imgs = [None for _ in range(len(labels))]

    def maybe_transform(self, x) -> np.ndarray:
        if self.transform is None:
            return x
        else:
            return self.transform(x)

    def load_image(self, idx):
        if self.in_memory:
            if self.cached_imgs[idx] is None:
                self.cached_imgs[idx] = self.maybe_transform(load_img(self.img_paths[idx]))

            return self.cached_imgs[idx]
        else:
            return self.maybe_transform(load_img(self.img_paths[idx]))

    def __getitem__(self, idx) -> Tuple[np.ndarray, int]:
        label = self.labels[idx]
        img = self.load_image(idx)

        return img, label

    def __len__(self) -> int:
        return len(self.img_paths)

    def filter_out_classes(self, classes_to_keep: Iterable[int]) -> "ImageDataset":
        classes_to_keep = set(classes_to_keep)
        idx_to_keep = [i for i, l in enumerate(self.labels) if l in classes_to_keep]

        return ImageDataset(
            [self.img_paths[i] for i in idx_to_keep],
            [self.labels[i] for i in idx_to_keep],
            self.transform,
            self.in_memory
        )

    def tolist(self) -> "ImageDataset":
        return [xy for xy in self]

    def get_subset(self, idx) -> "ImageDataset":
        return ImageDataset(
            [self.img_paths[i] for i in idx],
            [self.labels[i] for i in idx],
            self.transform,
            self.in_memory
        )
