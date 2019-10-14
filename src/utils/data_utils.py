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

    # splits = np.array([
    #     [87, 81, 139, 180, 163, 78, 70, 159, 124, 131],
    #     [61, 162, 136, 151, 186, 174, 36, 177, 118, 107],
    #     [160, 197, 39, 156, 127, 57, 93, 55, 153, 62],
    #     [69, 59, 187, 165, 95, 84, 41, 166, 88, 47],
    #     [133, 103, 185, 65, 50, 121, 134, 72, 141, 192],
    #     [66, 129, 140, 175, 91, 161, 89, 73, 155, 172],
    #     [96, 135, 122, 104, 54, 147, 38, 149, 52, 109],
    #     [125, 190, 171, 42, 191, 194, 167, 31, 189, 94],
    #     [82, 164, 98, 101, 51, 63, 30, 195, 123, 92],
    #     [181, 193, 108, 85, 176, 112, 34, 58, 40, 178],
    #     [45, 132, 144, 35, 97, 130, 37, 198, 179, 128],
    #     [76, 115, 48, 79, 199, 113, 120, 74, 116, 152],
    #     [100, 138, 86, 102, 71, 46, 75, 114, 119, 143],
    #     [126, 196, 43, 117, 111, 168, 90, 44, 137, 145],
    #     [170, 49, 32, 33, 183, 105, 158, 169, 64, 67],
    #     [106, 80, 157, 110, 99, 142, 150, 148, 56, 154],
    #     [188, 60, 184, 53, 173, 182, 146, 83, 68, 77]
    # ])

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
