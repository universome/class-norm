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
