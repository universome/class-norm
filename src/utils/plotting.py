from typing import Iterable

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(trainer):
    trainer.model.eval()

    n_samples = trainer.config.plotting.get('num_samples_per_class', 1)
    z = torch.tensor(trainer.fixed_noise[:n_samples]).to(trainer.device_name)

    for y in trainer.seen_classes:
        labels = torch.tensor([y] * n_samples).to(trainer.device_name).long()

        # TODO: separate classes in conditional batchnorm layer instead
        with torch.no_grad():
            x = trainer.model.generator(z, labels).cpu()

        imgs = ((x.permute(0, 2, 3, 1).numpy() + 1) * 127.5).astype(int)
        img_h, img_w = imgs.shape[1], imgs.shape[2]

        n_cols = 10
        n_rows = np.ceil(n_samples / n_cols).astype(int)

        result = np.zeros((n_rows * img_h, n_cols * img_w, 3)).astype(int)

        for i, img in enumerate(imgs):
            row_idx = i // n_cols
            col_idx = i % n_cols
            result[row_idx * img_h:(row_idx + 1) * img_h, col_idx * img_w:(col_idx + 1) * img_w] = img

        fig = plt.figure(figsize=(25, n_rows * 5))
        plt.imshow(result)

    trainer.writer.add_figure(f'Task_{y}', fig, trainer.num_iters_done)
