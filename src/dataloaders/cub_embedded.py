import os
from typing import List, Tuple

import numpy as np

def load_dataset(data_dir: str, is_train: bool=True) -> List[Tuple[np.ndarray, int]]:
    split = 'train' if is_train else 'test'

    return np.load(os.path.join(data_dir, f'{split}.npy'), allow_pickle=True)
