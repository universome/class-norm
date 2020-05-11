import random
import warnings
from typing import List, Tuple, Any

import numpy as np
from torch.utils.data import Subset
from skimage.transform import resize
from firelab.config import Config

from src.dataloaders.dataset import ImageDataset


def get_data_splits(class_splits: List[List[int]], dataset: ImageDataset) -> List[ImageDataset]:
    return [get_subset_by_labels(dataset, cls_group) for cls_group in class_splits]


def get_train_test_data_splits(class_splits:List[List[int]], ds_train: ImageDataset, ds_test: ImageDataset) -> List[Tuple[ImageDataset, ImageDataset]]:
    data_splits_train = get_data_splits(class_splits, ds_train)
    data_splits_test = get_data_splits(class_splits, ds_test)
    data_splits = list(zip(data_splits_train, data_splits_test))

    return data_splits


def get_subset_by_labels(dataset: ImageDataset, labels: List[int]) -> List[int]:
    """
    Finds objects with specific labels and returns them
    """
    subset_labels = set(labels)
    subset_idx = [i for i, y in enumerate(dataset.labels) if y in subset_labels]
    subset = Subset(dataset, subset_idx)

    return subset


def split_classes_for_tasks(config: Config) -> List[List[int]]:
    """
    Splits classes into `num_tasks` groups and returns these splits

    :param num_classes:
    :param num_tasks:
    :param num_classes_per_task:
    :return:
    """

    if config.has('task_sizes'):
        num_classes_to_use = sum(config.task_sizes)
    else:
        num_classes_to_use = config.num_tasks * config.num_classes_per_task

    if num_classes_to_use > config.num_classes:
        warnings.warn(f"We'll have duplicated classes: {num_classes_to_use} > {config.num_classes}")

    classes = np.arange(config.num_classes)
    classes = np.tile(classes, np.ceil(num_classes_to_use / len(classes)).astype(int))[:num_classes_to_use]
    classes = np.random.permutation(classes)

    # classes = np.array([1,2,4,6,9,10,11,12,14,15,16,17,18,19,20,21,23,24,25,26,27,29,31,38,39,40,41,43,44,45,46,47,49,51,53,54,55,56,57,58,59,60,61,62,63,64,66,67,68,69,70,72,73,74,75,76,77,79,80,81,84,86,87,88,89,91,92,93,96,98,99,103,104,105,106,107,108,109,110,112,114,115,116,117,119,121,122,123,124,125,126,127,128,130,131,132,133,135,136,138,139,140,141,142,143,144,145,147,148,149,150,151,152,153,154,156,157,158,159,160,161,163,166,167,168,169,170,171,172,173,174,175,176,177,178,180,181,183,187,188,189,190,191,192,193,194,195,197,198,199,0,3,5,7,8,13,22,28,30,32,33,34,35,36,37,42,48,50,52,65,71,78,82,83,85,90,94,95,97,100,101,102,111,113,118,120,129,134,137,146,155,162,164,165,179,182,184,185,186,196])

    if config.has('task_sizes'):
        steps = flatten([[0], np.cumsum(config.task_sizes[:-1])])
        splits = [classes[c:c + size].tolist() for c, size in zip(steps, config.task_sizes)]
    else:
        splits = classes.reshape(config.num_tasks, config.num_classes_per_task)
        splits = splits.tolist()


    return splits


def construct_output_mask(classes: List[int], total_num_classes:int) -> np.ndarray:
    """Returns 1D array of the output mask"""

    mask = np.zeros(total_num_classes).astype(bool)

    if len(classes) > 0:
        mask[classes] = True

    return mask


def resize_dataset(dataset: ImageDataset, w:int, h:int) -> ImageDataset:
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


def filter_out_classes(ds: List[Tuple[np.ndarray, int]], classes_to_keep: List[int]) -> List[Tuple[np.ndarray, int]]:
    """Removes datapoints with classes that are not in `classes_to_keep` list"""
    return [(x, y) for x, y in ds if y in classes_to_keep]


def flatten(list_of_lists: List[List[Any]]) -> List[Any]:
    return [x for list in list_of_lists for x in list]


def sample_instances_for_em(ds: List[Tuple[np.ndarray, int]], class_idx: int, size: int) -> List[Tuple[np.ndarray, int]]:
    class_samples = [(x, y) for x, y in ds if y == class_idx]
    memory = random.sample(class_samples, min(size, len(class_samples)))

    return memory
