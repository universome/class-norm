import os
import pickle
from os import PathLike
from typing import List, Tuple

import numpy as np

from .utils import read_column, shuffle_dataset, load_imgs, preprocess_imgs


def load_dataset(data_dir: PathLike, split: str='train', target_shape=(224, 224)) -> List[Tuple[np.ndarray, int]]:
    assert split in ['train', 'val', 'test'], f"Unknown dataset split: {split}"

    filename = os.path.join(data_dir, f'AWA_{split}_list.txt')
    img_paths = read_column(filename, 0)
    labels = read_column(filename, 1)
    labels = [int(l) for l in labels]

    # import random
    # chosen_idx = np.arange(len(img_paths)).tolist()
    # chosen_idx = random.sample(chosen_idx, 500)
    # img_paths = [img_paths[i] for i in chosen_idx]
    # labels = [labels[i] for i in chosen_idx]

    imgs = load_imgs(os.path.join(data_dir, 'Animals_with_Attributes2'), img_paths)
    imgs = preprocess_imgs(imgs, target_shape)
    imgs, labels = shuffle_dataset(imgs, labels)

    return list(zip(imgs, labels))


def load_class_attributes(data_dir: PathLike) -> np.ndarray:
    filename = os.path.join(data_dir, 'AWA_attr_in_order.pickle')

    with open(filename, 'rb') as f:
        attrs = pickle.load(f, encoding='latin1')

    return attrs
