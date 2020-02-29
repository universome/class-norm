from typing import Tuple
import numpy as np
from firelab.config import Config

from src.dataloaders import cub, feats, awa, svhn, mnist, cifar
from src.dataloaders.utils import extract_resnet_features_for_dataset


SIMPLE_LOADERS = {
    'SVHN': svhn.load_dataset,
    'MNIST': mnist.load_dataset,
    'CIFAR10': lambda *args, **kwargs: cifar.load_dataset(*args, **kwargs, num_classes=10),
    'CIFAR100': lambda *args, **kwargs: cifar.load_dataset(*args, **kwargs, num_classes=100),
}

def load_data(config: Config, img_target_shape: Tuple[int, int]=None,
              preprocess: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.name == 'CUB':
        ds_train = cub.load_dataset(config.dir, split='train', target_shape=img_target_shape, preprocess=preprocess)
        ds_test = cub.load_dataset(config.dir, split='test', target_shape=img_target_shape, preprocess=preprocess)
        class_attributes = cub.load_class_attributes(config.dir).astype(np.float32)
    elif config.name == 'CUB_EMBEDDINGS':
        ds_train = feats.load_dataset(config.dir, config.input_type, split='train')
        ds_test = feats.load_dataset(config.dir, config.input_type, split='test')
        class_attributes = cub.load_class_attributes(config.dir).astype(np.float32)
    elif config.name == 'AWA':
        ds_train = awa.load_dataset(config.dir, split='train', target_shape=img_target_shape, preprocess=preprocess)
        ds_test = awa.load_dataset(config.dir, split='test', target_shape=img_target_shape, preprocess=preprocess)
        class_attributes = awa.load_class_attributes(config.dir).astype(np.float32)
    elif config.name in SIMPLE_LOADERS.keys():
        ds_train = SIMPLE_LOADERS[config.name](config.dir, split='train')
        ds_test = SIMPLE_LOADERS[config.name](config.dir, split='test')
        class_attributes = None
    elif config.name.endswith('EMBEDDINGS'):
        ds_train = feats.load_dataset(config.dir, config.input_type, split='train')
        ds_test = feats.load_dataset(config.dir, config.input_type, split='test')
        class_attributes = None
    else:
        raise NotImplementedError(f'Unkown dataset: {config.name}')

    # if embed_data:
    #     ds_train = extract_resnet_features_for_dataset(ds_train, input_type=18)
    #     ds_test = extract_resnet_features_for_dataset(ds_test, input_type=18)

    # np.save(f'/tmp/{config.name}_train', ds_train)
    # np.save(f'/tmp/{config.name}_test', ds_test)
    # ds_train = np.load(f'/tmp/{config.name}_train.npy', allow_pickle=True)
    # ds_test = np.load(f'/tmp/{config.name}_test.npy', allow_pickle=True)

    return ds_train, ds_test, class_attributes
