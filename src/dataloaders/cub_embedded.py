import os
from typing import List, Tuple

import numpy as np


def load_dataset(data_dir: str, input_type: str, split: str) -> List[Tuple[np.ndarray, int]]:
    file_name = f'{split}_{input_type}.npy'
    print(f'Loading {file_name}')
    data = np.load(os.path.join(data_dir, file_name), allow_pickle=True)

    return data
