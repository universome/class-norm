import os
from typing import List, Tuple

import numpy as np

def load_dataset(data_dir: str, resnet_type: int=18, feat_level: str='fc', is_train: bool=True) -> List[Tuple[np.ndarray, int]]:
    split = 'train' if is_train else 'test'
    feat_level_prefix = '' if feat_level == 'fc' else 'conv_'
    file_name = f'{split}_{feat_level_prefix}feats_resnet{resnet_type}_tiny.npy'
    print(f'Loading {file_name}')
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True)

    return data
