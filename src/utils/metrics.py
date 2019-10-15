import copy
from typing import List
import numpy as np


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
    accuracies_history = np.array(accuracies_history)
    after_task_num = (after_task_idx + 1) if after_task_idx >= 0 else len(accuracies_history)
    prev_accs = accuracies_history[:after_task_num - 1, :after_task_num - 1]
    forgettings = prev_accs.max(axis=0) - accuracies_history[after_task_num - 1, :after_task_num - 1]
    forgetting_measure = np.mean(forgettings).item()

    return forgetting_measure


def compute_learning_curve_area(accs:List[List[float]], beta: int=10) -> float:
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


def compute_ausuc(logits: List[List[float]], targets: List[int], seen_classes_mask: List[int]) -> float:
    """
    Computes area under Seen-Unseen curve (https://arxiv.org/abs/1605.04253)

    :param logits: predicted logits of size [DATASET_SIZE x NUM_CLASSES]
    :param targets: targets of size [DATASET_SIZE]
    :param seen_classes_mask: mask, indicating seen classes of size [NUM_CLASSES]

    :return: AUSUC metric
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

    if len(seen_classes) == 0: return (logits_unseen.argmax(axis=1) == targets_unseen).mean()
    if len(unseen_classes) == 0: return (logits_seen.argmax(axis=1) == targets_seen).mean()

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

    return auc_score


def compute_ausuc_slow(logits: List[List[float]], targets: List[int], seen_classes_mask: List[int]) -> float:
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

    for GZSL_lambda in np.arange(-10, 10, 0.01):
        tmp_seen_sim = copy.deepcopy(logits_on_seen_ds)
        tmp_seen_sim[:, unseen_classes] += GZSL_lambda
        acc_S_T_list.append((tmp_seen_sim.argmax(axis=1) == targets_on_seen_ds).mean())

        tmp_unseen_sim = copy.deepcopy(logits_on_unseen_ds)
        tmp_unseen_sim[:, unseen_classes] += GZSL_lambda
        acc_U_T_list.append((tmp_unseen_sim.argmax(axis=1) == targets_on_unseen_ds).mean())

    return np.trapz(y=acc_S_T_list, x=acc_U_T_list) * 100.0
