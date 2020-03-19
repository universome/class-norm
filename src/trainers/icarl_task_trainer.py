import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F

from src.trainers.task_trainer import TaskTrainer
from src.dataloaders.utils import extract_features
from src.utils.data_utils import flatten
from src.utils.training_utils import compute_accuracy, prune_logits


class iCarlTaskTrainer(TaskTrainer):
    def train_on_batch(self, batch):
        self.model.train()

        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        y = torch.from_numpy(np.array(batch[1])).to(self.device_name)

        logits = self.model(x)
        pruned_logits = prune_logits(logits, self.output_mask)

        cls_loss = F.cross_entropy(pruned_logits, y)
        cls_acc = compute_accuracy(pruned_logits, y)

        total_loss = cls_loss

        if self.task_idx > 0:
            rehearsal_loss, rehearsal_acc = self.compute_rehearsal_loss()
            total_loss += self.config.hp.memory.loss_coef * rehearsal_loss

            self.writer.add_scalar('train/rehearsal_loss', rehearsal_loss.item(), self.num_iters_done)
            self.writer.add_scalar('train/rehearsal_acc', rehearsal_acc.item(), self.num_iters_done)

        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        self.writer.add_scalar('train/cls_loss', cls_loss.item(), self.num_iters_done)
        self.writer.add_scalar('train/cls_acc', cls_acc.item(), self.num_iters_done)

    def compute_rehearsal_loss(self):
        x, y = self.sample_from_memory(self.config.hp.memory.batch_size)
        pruned_logits = prune_logits(self.model(x), self.learned_classes_mask)
        cls_loss = F.cross_entropy(pruned_logits, y)
        cls_acc = compute_accuracy(pruned_logits, y)

        return cls_loss, cls_acc

    def update_episodic_memory(self):
        num_samples_per_class = int(np.ceil(self.config.hp.memory.max_size / len(self.seen_classes)))

        self.reduce_episodic_memory(num_samples_per_class)
        self.extend_episodic_memory(num_samples_per_class)

    def reduce_episodic_memory(self, num_samples_per_class: int):
        class_memories = [[(x, y) for x, y in self.episodic_memory if y == c] for c in self.learned_classes]
        class_memories_reduced = [mem[:num_samples_per_class] for mem in class_memories]

        self.episodic_memory = flatten(class_memories_reduced)

    def extend_episodic_memory(self, num_samples_per_class: int):
        for c in self.classes:
            selected_idx = []
            imgs = [x for (x, y) in self.load_dataset(self.task_ds_train) if y == c]
            feats = torch.from_numpy(np.array(extract_features(imgs, self.model.embedder, 256, verbose=False)))
            feats = feats / feats.norm(dim=1, keepdim=True)
            prototype_gold = torch.from_numpy(np.array(feats).mean(axis=0))
            prototype_gold = prototype_gold / prototype_gold.norm()
            feats_selected = torch.empty(0, len(prototype_gold))
            feats_remaining = feats

            while len(selected_idx) < num_samples_per_class and len(feats_remaining) > 0:
                assert len(selected_idx) == len(feats_selected)
                assert (len(feats_selected) + len(feats_remaining)) == len(imgs)

                best_idx = self.select_best_current_idx(feats_remaining, feats_selected, prototype_gold)
                selected_idx.append(best_idx)
                feats_selected = torch.cat([feats_selected, feats_remaining[best_idx].unsqueeze(0)])
                feats_remaining = torch.cat([feats_remaining[:best_idx], feats_remaining[best_idx + 1:]])

            self.episodic_memory.extend([(imgs[i], c) for i in selected_idx])

    def select_best_current_idx(self, feats_remaining: Tensor, feats_selected: Tensor,
                                prototype_gold: Tensor) -> int:
        prototypes_next = (feats_selected.sum(dim=0, keepdim=True) + feats_remaining) / (len(feats_selected) + 1)
        prototypes_next = prototypes_next / prototypes_next.norm(dim=1, keepdim=True)
        distances = (prototypes_next - prototype_gold.unsqueeze(0)).pow(2).sum(dim=1)
        best_idx = distances.argmax(dim=0)

        return best_idx
