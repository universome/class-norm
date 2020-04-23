import copy
from typing import List, Tuple
import numpy as np

from src.utils.data_utils import flatten, construct_output_mask


def compute_average_accuracy(accuracies_history: List[List[float]], after_task_idx: int=-1) -> float:
    """
    Computes average accuracy

    :param accuracies_history: matrix of size [NUM_TASKS x NUM_TASKS]
    :return: average accuracy after the specified task
    """
    accuracies_history = np.array(accuracies_history)
    after_task_num = (after_task_idx + 1) if after_task_idx >= 0 else len(accuracies_history)
    accs_after_the_task = accuracies_history[after_task_num - 1, :after_task_num]
    avg_acc = np.mean(accs_after_the_task).item()

    return avg_acc


def compute_forgetting_measure(accuracies_history: List[List[float]], after_task_idx: int=-1) -> float:
    """
    Computes forgetting measure after specified task.
    Not that it is computed for all tasks except the last one

    :param accuracies_history: matrix of size [NUM_TASKS x NUM_TASKS]
    :param after_task_idx: up to which task (non-inclusive) to compute the metric
    :return: forgetting measure
    """
    n_tasks = accuracies_history.shape[0]
    assert accuracies_history.shape == (n_tasks, n_tasks), f"Wrong shape: {accuracies_history.shape}"
    assert after_task_idx == -1 or after_task_idx > 0

    accuracies_history = np.array(accuracies_history)
    after_task_num = (after_task_idx + 1) if after_task_idx >= 0 else len(accuracies_history)
    prev_accs = accuracies_history[:after_task_num - 1, :after_task_num - 1]
    forgettings = prev_accs.max(axis=0) - accuracies_history[after_task_num - 1, :after_task_num - 1]
    forgetting_measure = np.mean(forgettings).item()

    return forgetting_measure


def compute_learning_curve_area(accs: List[List[float]], beta: int=10) -> float:
    """
    Comptues learning curve area for a specific value of beta

    :param accs: array of accuracy histories for each task after each training batch
                It should has size [NUM_TASKS] x [NUM_BATCHES + 1],
                where NUM_BATCHES can be different for different tasks
    :param beta: how many batches to take into account
    :return: LCA metric
    """
    assert all([(len(task_accs) >= beta + 1) for task_accs in accs])

    accs = np.array([task_accs[:beta + 1] for task_accs in accs])
    lca = np.mean(accs).item()

    return lca


def compute_ausuc(logits: List[List[float]], targets: List[int], seen_classes_mask: List[bool], return_accs: bool=False) -> Tuple[float, Tuple[List[float], List[float]]]:
    """
    Computes area under Seen-Unseen curve (https://arxiv.org/abs/1605.04253)

    :param logits: predicted logits of size [DATASET_SIZE x NUM_CLASSES]
    :param targets: targets of size [DATASET_SIZE]
    :param seen_classes_mask: mask, indicating seen classes of size [NUM_CLASSES]

    :return: AUSUC metric and corresponding curve values
    """

    logits = np.array(logits)
    targets = np.array(targets)
    seen_classes_mask = np.array(seen_classes_mask)
    ds_size, num_classes = logits.shape

    assert len(targets) == ds_size
    assert len(seen_classes_mask) == num_classes

    seen_classes = np.nonzero(seen_classes_mask)[0]
    unseen_classes = np.nonzero(~seen_classes_mask)[0]

    logits_seen = logits[:, seen_classes]
    logits_unseen = logits[:, unseen_classes]

    targets_seen = np.array([next((i for i, t in enumerate(seen_classes) if y == t), -1) for y in targets])
    targets_unseen = np.array([next((i for i, t in enumerate(unseen_classes) if y == t), -1) for y in targets])

    if len(seen_classes) == 0:
        acc = (logits_unseen.argmax(axis=1) == targets_unseen).mean()
        accs_seen = np.array([1., 1., 0.])
        accs_unseen = np.array([0., acc, acc])
    elif len(unseen_classes) == 0:
        acc = (logits_seen.argmax(axis=1) == targets_seen).mean()
        accs_seen = np.array([acc, acc, 0.])
        accs_unseen = np.array([0., 1., 1.])
    else:
        gaps = logits_seen.max(axis=1) - logits_unseen.max(axis=1)
        sorting = np.argsort(gaps)[::-1]
        guessed_seen = logits_seen[sorting].argmax(axis=1) == targets_seen[sorting]
        guessed_unseen = logits_unseen[sorting].argmax(axis=1) == targets_unseen[sorting]

        accs_seen = np.cumsum(guessed_seen) / (targets_seen != -1).sum()
        accs_unseen = np.cumsum(guessed_unseen) / (targets_unseen != -1).sum()
        accs_unseen = accs_unseen[-1] - accs_unseen

        accs_seen = accs_seen[::-1]
        accs_unseen = accs_unseen[::-1]

    auc_score = np.trapz(accs_seen, x=accs_unseen) * 100

    if return_accs:
        return auc_score, (accs_seen, accs_unseen)
    else:
        return auc_score


def compute_ausuc_slow(logits: List[List[float]], targets: List[int], seen_classes_mask: List[bool],
                       lambda_range=np.arange(-10, 10, 0.01)) -> float:
    targets = np.array(targets)
    logits = np.array(logits)
    seen_classes_mask = np.array(seen_classes_mask)

    acc_S_T_list, acc_U_T_list = list(), list()
    seen_classes = np.nonzero(seen_classes_mask)[0]
    unseen_classes = np.nonzero(~seen_classes_mask)[0]
    logits_on_seen_ds = logits[[y in seen_classes for y in targets]]
    logits_on_unseen_ds = logits[[y in unseen_classes for y in targets]]
    targets_on_seen_ds = targets[[y in seen_classes for y in targets]]
    targets_on_unseen_ds = targets[[y in unseen_classes for y in targets]]

    for GZSL_lambda in lambda_range:
        tmp_seen_sim = copy.deepcopy(logits_on_seen_ds)
        tmp_seen_sim[:, unseen_classes] += GZSL_lambda
        acc_S_T_list.append((tmp_seen_sim.argmax(axis=1) == targets_on_seen_ds).mean())

        tmp_unseen_sim = copy.deepcopy(logits_on_unseen_ds)
        tmp_unseen_sim[:, unseen_classes] += GZSL_lambda
        acc_U_T_list.append((tmp_unseen_sim.argmax(axis=1) == targets_on_unseen_ds).mean())

    return np.trapz(y=acc_S_T_list, x=acc_U_T_list) * 100.0


def compute_ausuc_matrix(logits_history: np.ndarray, targets: List[int], class_splits: List[List[int]]) -> np.ndarray:
    """
    Computes pairwise AUSUC scores between tasks given logits history

    :param logits_history: history of model logits, evaluated BEFORE each task,
                           i.e. matrix of size [NUM_TASKS x DATASET_SIZE x NUM_CLASSES]
    :param targets: targets for the objects of size [DATASET_SIZE]
    :param class_splits: list of classes for each task of size [NUM_TASKS x NUM_CLASSES_PER_TASK]

    :return: AUCSUC value and a matrix of pairwise AUSUCS
    """
    num_tasks = len(logits_history)
    ausuc_matrix = []

    for task_from in range(num_tasks):
        ausucs = []

        for task_to in range(num_tasks):
            classes = set(flatten([class_splits[task_to], class_splits[task_from]]))
            curr_logits = [l for l, t in zip(logits_history[task_from], targets) if t in classes]
            curr_targets = [t for t in targets if t in classes]

            classes = list(classes)
            curr_targets = remap_targets(curr_targets, classes)
            seen_classes_mask = np.array([c in class_splits[task_from] for c in classes]).astype(bool)
            ausuc, _ = compute_ausuc(np.array(curr_logits)[:, classes], curr_targets, seen_classes_mask)
            ausucs.append(ausuc)

        ausuc_matrix.append(ausucs)

    return np.array(ausuc_matrix)


def compute_individual_accs_matrix(logits_history: np.ndarray, targets: List[int], class_splits: List[List[int]], restrict_space: bool=False) -> np.ndarray:
    """
    Computes accuracy for each task for each timestep.
    You would like to use np.triu or np.triu_indices to get zero-shot accuracies

    :param logits_history: history of model logits, evaluated BEFORE each task,
                           i.e. matrix of size [NUM_TASKS x DATASET_SIZE x NUM_CLASSES]
    :param targets: targets for the objects of size [DATASET_SIZE]
    :param class_splits: list of classes for each task of size [NUM_TASKS x NUM_CLASSES_PER_TASK]

    :return: matrix of accuracies of size NUM_TASKS x NUM_TASKS
    """

    # TODO: actually, we do a lot of computations that can be cached here :|
    return np.array([[compute_acc_for_classes(l, targets, cs, restrict_space=restrict_space) for cs in class_splits] for l in logits_history])


def compute_task_transfer_matrix(logits_history: np.ndarray, targets: List[int], class_splits: List[List[int]]) -> np.ndarray:
    """
    Task transfer matrix is a matrix of accuracy differences in each task.
    It depicts changes in performance for each task at each timestep.

    :param logits_history: history of model logits, evaluated at moments 0, 1, ..., T,
                           i.e. matrix of size [(NUM_TASKS + 1) x DATASET_SIZE x NUM_CLASSES]
    :param targets: targets for the objects of size [DATASET_SIZE]
    :param class_splits: list of classes for each task of size [NUM_TASKS x NUM_CLASSES_PER_TASK]

    :return: task transfer matrix of size NUM_TASKS x NUM_TASKS
    """
    accs_before = compute_individual_accs_matrix(logits_history[:-1], targets, class_splits)
    accs_after = compute_individual_accs_matrix(logits_history[1:], targets, class_splits)

    return accs_after - accs_before


def compute_unseen_classes_acc_history(logits_history: List[List[List[float]]], targets: List[int],
                                   class_splits: List[List[int]], restrict_space:bool=True) -> List[float]:
    """
    Computes zero-shot history on all the remaining tasks before starting each task

    :param logits_history: history of model logits, evaluated BEFORE each task,
                           i.e. matrix of size [NUM_TASKS x DATASET_SIZE x NUM_CLASSES]
    :param targets: targets for the objects of size [DATASET_SIZE]
    :param class_splits: list of classes for each task of size [NUM_TASKS x NUM_CLASSES_PER_TASK]
    :param restrict_space: should we restrict prediction space to specified classes or not

    :return: zero-shot accuracies of size [NUM_TASKS]
    """
    unseen_classes = [np.unique(flatten(class_splits[i:])) for i in range(len(class_splits))]
    accs = [compute_acc_for_classes(l, targets, cs, restrict_space) for l, cs in zip(logits_history, unseen_classes)]

    return accs


def compute_seen_classes_acc_history(logits_history: List[List[List[float]]], targets: List[int],
                                     class_splits: List[List[int]], restrict_space:bool=True) -> List[float]:
    """
    Computes zero-shot history on all the remaining tasks before starting each task

    :param logits_history: history of model logits, evaluated AFTER each task,
                           i.e. matrix of size [NUM_TASKS x DATASET_SIZE x NUM_CLASSES]
    :param targets: targets for the objects of size [DATASET_SIZE]
    :param class_splits: list of classes for each task of size [NUM_TASKS x NUM_CLASSES_PER_TASK]
    :param restrict_space: should we restrict prediction space to specified classes or not

    :return: zero-shot accuracies of size [NUM_TASKS]
    """
    seen_classes = [np.unique(flatten(class_splits[:i+1])) for i in range(len(class_splits))]
    accs = [compute_acc_for_classes(l, targets, cs, restrict_space) for l, cs in zip(logits_history, seen_classes)]

    return accs


def compute_joined_ausuc_history(logits_history: List[List[List[float]]], targets: List[int],
                                 class_splits: List[List[int]]) -> List[float]:
    """
    Computes AUSUC history on all the remaining tasks before starting each task

    :param logits_history: history of model logits, evaluated BEFORE each task,
                           i.e. matrix of size [NUM_TASKS x DATASET_SIZE x NUM_CLASSES]
    :param targets: targets for the objects of size [DATASET_SIZE]
    :param class_splits: list of classes for each task of size [NUM_TASKS x NUM_CLASSES_PER_TASK]

    :return: AUSUC scores of size [NUM_TASKS]
    """
    num_classes = len(logits_history[0][0])
    seen_classes = [np.unique(flatten(class_splits[:i])) for i in range(len(class_splits))]
    seen_classes_masks = [construct_output_mask(cs, num_classes) for cs in seen_classes]
    ausuc_scores = [compute_ausuc(l, targets, m)[0] for l, m in zip(logits_history, seen_classes_masks)]

    return ausuc_scores


def compute_acc_for_classes(logits: List[List[float]], targets: List[int],
                            classes: List[int], restrict_space:bool=True) -> float:
    """
    Computes accuracy for a given classes, i.e. we prune out all the other classes

    :param logits: matrix of size [DATASET_SIZE x NUM_CLASSES]
    :param targets: targets for the objects of size [DATASET_SIZE]
    :param classes: list of classes to consider
    :param restrict_space: should we restrict prediction space to specified classes or not

    :return: accuracy
    """
    data_idx = [i for i, t in enumerate(targets) if t in classes]
    targets = np.array(targets)[data_idx]
    logits = np.array(logits)[data_idx]

    if restrict_space:
        targets = np.array(remap_targets(targets, list(classes)))
        logits = logits[:, classes]

    acc = (logits.argmax(axis=1) == targets).mean()

    return acc


def remap_targets(targets: List[int], classes: List[int]) -> List[int]:
    """
    Takes target classes and remaps them into a smaller range, determined by classes argument

    :param targets: dataset targets, vector if length [DATASET_SIZE]
    :param classes: classes to map
    :return: remapped classes
    """
    return [(classes.index(t) if t in classes else -1) for t in targets]
