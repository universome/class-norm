import os
from typing import List, Tuple

import numpy as np

def load_dataset(data_dir: str, input_type: str='resnet18_feat', is_train: bool=True) -> List[Tuple[np.ndarray, int]]:
    split = 'train' if is_train else 'test'
    file_name = f'{split}_{input_type}.npy'
    print(f'Loading {file_name}')
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True)

    return data
