from typing import Tuple
import numpy as np
from firelab.config import Config

from src.dataloaders import cub, awa, svhn
from src.dataloaders.utils import extract_resnet18_features_for_dataset


def load_data(config: Config, img_target_shape: Tuple[int, int], embed_data: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.name == 'CUB':
        ds_train = cub.load_dataset(config.dir, is_train=True, target_shape=img_target_shape)
        ds_test = cub.load_dataset(config.dir, is_train=False, target_shape=img_target_shape)
        class_attributes = cub.load_class_attributes(config.dir).astype(np.float32)
    elif config.name == 'AWA':
        ds_train = awa.load_dataset(config.dir, split='train', target_shape=img_target_shape)
        ds_test = awa.load_dataset(config.dir, split='test', target_shape=img_target_shape)
        class_attributes = awa.load_class_attributes(config.dir).astype(np.float32)
    elif config.name == 'SVHN':
        ds_train = svhn.load_dataset(config.dir, split='train', target_shape=img_target_shape)
        ds_test = svhn.load_dataset(config.dir, split='test', target_shape=img_target_shape)
        class_attributes = None
    else:
        raise NotImplementedError(f'Unkown dataset: {config.name}')

    if embed_data:
        ds_train = extract_resnet18_features_for_dataset(ds_train)
        ds_test = extract_resnet18_features_for_dataset(ds_test)

    # np.save(f'/tmp/{config.name}_train', ds_train)
    # np.save(f'/tmp/{config.name}_test', ds_test)
    # ds_train = np.load(f'/tmp/{config.name}_train.npy', allow_pickle=True)
    # ds_test = np.load(f'/tmp/{config.name}_test.npy', allow_pickle=True)

    return ds_train, ds_test, class_attributes
