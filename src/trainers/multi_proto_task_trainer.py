from scipy.optimize import fsolve, bisect
import numpy as np
import torch
from torch import Tensor

from src.trainers.task_trainer import TaskTrainer
from src.utils.training_utils import prune_logits
from src.utils.losses import compute_mean_distance


class MultiProtoTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        logits, protos = self.model(x, return_protos=True)

        with torch.no_grad():
            probs = logits[:, self.output_mask].softmax(dim=1)
            entropy = np.array([entropy_for_logits(ls) for ls in logits[:, self.output_mask].cpu().numpy()])

            self.writer.add_scalar('mean_entropy', entropy.mean(), self.num_iters_done)
            self.writer.add_scalar('mean_max_prob', probs.max(dim=1)[0].mean().item(), self.num_iters_done)
            self.writer.add_scalar('mean_min_prob', probs.min(dim=1)[0].mean().item(), self.num_iters_done)

        scales = compute_optimal_temperature(logits, self.config.hp.head.optimal_entropy)
        self.writer.add_scalar('mean_scale', scales.mean().item(), self.num_iters_done)
        scales = scales.clamp_max(25)
        logits = logits * scales.unsqueeze(1)

        with torch.no_grad():
            entropy = np.array([entropy_for_logits(ls) for ls in logits[:, self.output_mask].cpu().numpy()])
            self.writer.add_scalar('mean_entropy_after', entropy.mean(), self.num_iters_done)

        pruned = prune_logits(logits, self.output_mask)
        cls_loss = self.criterion(pruned, y)
        loss = cls_loss

        if self.config.hp.push_protos_apart.enabled:
            mean_distance = compute_mean_distance(protos)
            loss += self.config.hp.push_protos_apart.loss_coef * (-1 * mean_distance)

            self.writer.add_scalar('mean_distance', mean_distance.item(), self.num_iters_done)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar('accuracy', (pruned.argmax(dim=1) == y).float().mean().item(), self.num_iters_done)


def compute_optimal_temperature(logits: Tensor, target_entropy_val: float) -> Tensor:
    logits_np = logits.detach().cpu().numpy()
    optimal_scalers = []

    # Finding sequentially (for now)
    for ls in logits_np:
        def func(scale: float):
            return entropy_for_logits(ls, scale) - target_entropy_val

        optimal_scalers.append(bisect(func, 1e-12, 1e+12))

    return torch.tensor(optimal_scalers).to(logits.device)


def entropy_for_logits(logits, *args):
    return entropy(softmax(logits, *args))


def entropy(probs):
    probs = np.array(probs)[np.array(probs) != 0]
    return -(np.log(probs) * probs).sum()


def softmax(values, scale=1):
    raw_log_probs = np.array(values) * scale
    raw_log_probs = raw_log_probs - raw_log_probs.max()
    raw_probs = np.exp(raw_log_probs)
    probs = raw_probs / raw_probs.sum()

    return probs


def linear_softmax(values, scale=1):
    log_probs = values - values.min()
    probs = log_probs / log_probs.sum()

    return probs
