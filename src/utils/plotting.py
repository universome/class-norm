import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(trainer):
    z = torch.tensor(trainer.fixed_noise[:trainer.config.plotting.num_samples_per_class]).to(trainer.device_name)

    for task_idx in range(trainer.task_idx):
        classes = trainer.main_trainer.class_splits[task_idx]
        trainer.plot_samples(task_idx, classes, z)

def plot_task_samples(trainer, task_idx: int, classes: Iterable[int], z: Tensor):
    for y in classes:
        y = np.array(y).tile(trainer.config.plotting.num_samples_per_class)
        y = torch.tensor(y).to(trainer.device_name)

        # TODO: separate classes in conditional batchnorm layer instead
        with torch.no_grad():
            classes = torch.tensor(np.ones(len(z)) * y).to(trainer.device_name).long()
            x = trainer.model.generator(z, classes).cpu()

        imgs = ((x.permute(0, 2, 3, 1).numpy() + 1) * 127.5).astype(int)
        img_h, img_w = imgs.shape[1], imgs.shape[2]

        n_rows = int(np.sqrt(trainer.config.plotting.num_samples_per_class))
        n_cols = n_rows

        assert n_rows * n_cols == trainer.config.plotting.num_samples_per_class

        result = np.zeros((n_rows * img_h, n_cols * img_w, 3)).astype(int)

        for i, img in enumerate(imgs):
            h = img_h * (i // n_rows)
            w = img_w * (i % n_cols)
            result[h:h + img_h, w:w + img_w] = img

        fig = plt.figure(figsize=(25, 25))
        plt.imshow(result)

    trainer.writer.add_figure(f'Task_{y}', fig, trainer.num_iters_done)
