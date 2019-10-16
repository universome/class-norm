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


def split_classes_for_tasks(num_classes: int, num_tasks: int, num_classes_per_task: int, num_reserved_classes: int=0) -> List[List[int]]:
    """
    Splits classes into `num_tasks` groups and returns these splits

    :param num_classes:
    :param num_tasks:
    :param num_classes_per_task:
    :param num_reserved_classes: — if we run live HPO, then we would like to reserve some of the first classes for it
    :return:
    """
    classes = np.arange(num_tasks * num_classes_per_task)
    classes = classes[num_reserved_classes:]
    splits = np.random.permutation(classes).reshape(num_tasks, num_classes_per_task)

    if num_tasks * num_classes_per_task > num_classes:
        warnings.warn('We will have duplicated classes')
        splits %= num_classes # Let's do this via module to have as less duplications as possbile

    return splits


def get_subset_by_labels(dataset, labels: List[int]) -> List[int]:
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


def compute_class_centroids(dataset: List[Tuple[np.ndarray, int]], total_num_classes: int) -> np.ndarray:
    """
    Computes class centroids and returns a matrix of size [TOTAL_NUM_CLASSES x X_DIM]
    :param imgs:
    :param labels:
    :return:
    """
    assert np.array(dataset[0][0]).ndim == 1, "We should work in features space instead of image space"

    centroids = np.zeros((total_num_classes, len(dataset[0][0])))
    unique_labels = list(set(y for _, y in dataset))
    centroids[unique_labels] = [np.mean([x for x, y in dataset if y == l], axis=0) for l in unique_labels] # TODO: this can be done much faster

    return centroids
