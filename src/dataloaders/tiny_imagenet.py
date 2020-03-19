import os
import random
from os import PathLike
from typing import List, Tuple

import numpy as np

from src.dataloaders.utils import shuffle_dataset, load_imgs, load_imgs_from_folder, preprocess_imgs, read_column

DEBUG = False
# DEBUG = True

def load_dataset(
        data_dir: PathLike,
        split: str='train',
        preprocess: bool=False,
        target_shape: Tuple[int, int]=None) -> List[Tuple[np.ndarray, int]]:

    assert split in ['train', 'val', 'test'], f"Unknown dataset split: {split}"

    with open(f'{data_dir}/wnids.txt', 'r') as f:
        class_names = f.read().splitlines()
        cls2idx = {c: i for i, c in enumerate(class_names)}

    if split == 'train':
        classes_dir = f'{data_dir}/train'
        classes = [d for d in os.listdir(classes_dir) if d.startswith('n')]
        classes_dirs = [f'{classes_dir}/{c}/images' for c in classes]
        classes_dirs = [d for d in classes_dirs if os.path.isdir(d)]
        all_imgs_paths = [f'{d}/{f}' for d in classes_dirs for f in os.listdir(d) if f.endswith('.JPEG')]
        assert len(all_imgs_paths) == 200 * 500

        if DEBUG:
            all_imgs_paths = [f'{d}/{f}' for d in classes_dirs for f in os.listdir(d)[:2] if f.endswith('.JPEG')]

        imgs = load_imgs(all_imgs_paths, target_shape)
        labels = [cls2idx[os.path.basename(p)[:9]] for p in all_imgs_paths]
    elif split == 'val':
        imgs_dir = f'{data_dir}/val/images'
        annotations_path = f'{data_dir}/val/val_annotations.txt'

        img_names = read_column(annotations_path, 0, sep='\t')
        class_names = read_column(annotations_path, 1, sep='\t')
        assert len(img_names) == len(class_names) == 200 * 50

        if DEBUG:
            img_names = img_names[:200]
            class_names = class_names[:200]

        imgs = load_imgs_from_folder(imgs_dir, img_names, target_shape)
        labels = [cls2idx[l] for l in class_names]
    else:
        raise NotImplementedError('We do not have labels for test split')

    if split == 'train': imgs, labels = shuffle_dataset(imgs, labels)
    if preprocess: imgs = preprocess_imgs(imgs)

    return list(zip(imgs, labels))
