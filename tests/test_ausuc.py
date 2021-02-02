import sys; sys.path.append('.')
from time import time

from tqdm import tqdm
import numpy as np

from src.utils.metrics import compute_ausuc, compute_ausuc_slow


def test_ausuc_on_random_data():
    for _ in tqdm(range(10), desc='Testing AUSUC'):
        num_classes = 100
        ds_size = 500
        logits = np.random.rand(ds_size, num_classes) * 5 - 2.5
        targets = np.random.randint(low=0, high=100, size=ds_size)
        logits[np.arange(ds_size), targets] += 1
        seen_classes_mask = np.random.rand(num_classes) > 0.5

        #start = time()
        ausuc_slow = compute_ausuc_slow(logits, targets, seen_classes_mask)
        #print(f'Slow: ({time() - start:.03f})', ausuc_slow)

        #start = time()
        ausuc_fast = compute_ausuc(logits, targets, seen_classes_mask)
        #print(f'Fast: ({time() - start:.03f})', ausuc_fast)

        assert np.abs(ausuc_slow - ausuc_fast) < 1e-2


def test_ausuc_on_edge_cases():
    num_classes = 100
    ds_size = 500
    logits = np.random.rand(ds_size, num_classes) * 5 - 2.5
    targets = np.random.randint(low=0, high=100, size=ds_size)
    logits[np.arange(ds_size), targets] += 1
    acc = (logits.argmax(axis=1) == targets).mean()

    assert np.abs(compute_ausuc(logits, targets, np.ones(num_classes).astype(bool)) - acc * 100) < 1e-6
    assert np.abs(compute_ausuc(logits, targets, np.zeros(num_classes).astype(bool)) - acc * 100) < 1e-6
