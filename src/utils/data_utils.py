import warnings
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize


def get_data_splits(class_splits:List[List[int]], dataset) -> List[Dataset]:
    return [get_subset_by_labels(dataset, cls_group) for cls_group in class_splits]


def get_train_test_data_splits(class_splits:List[List[int]], ds_train:Dataset, ds_test:Dataset) -> List[Tuple[Dataset, Dataset]]:
    data_splits_train = get_data_splits(class_splits, ds_train)
    data_splits_test = get_data_splits(class_splits, ds_test)
    data_splits = list(zip(data_splits_train, data_splits_test))

    return data_splits


def split_classes_for_tasks(num_classes: int, num_tasks: int) -> List[List[int]]:
    """
    Splits classes into `num_tasks` groups and returns these splits
    """
    if not num_classes % num_tasks != 0:
        warnings.warn(f'{num_classes} % {num_tasks} != 0. There are unused classes.')

    num_classes_per_task = num_classes // num_tasks
    num_classes_to_use = num_tasks * num_classes_per_task
    splits = np.random.permutation(num_classes_to_use).reshape(num_tasks, num_classes_per_task)

    return splits


def get_subset_by_labels(dataset, labels:List[int]) -> Dataset:
    """
    Finds objects with specific labels and returns them
    """
    labels = set(labels)
    subset = [(x,y) for x,y in dataset if y in labels]

    return subset


def construct_output_mask(task_labels:List[int], total_num_classes:int) -> np.ndarray:
    """Returns 1D array of the output mask"""

    mask = np.zeros(total_num_classes).astype(bool)
    mask[task_labels] = True

    return mask


def resize_dataset(dataset:Dataset, w:int, h:int) -> Dataset:
    return [(resize(x, (w, h)), y) for x, y in dataset]

