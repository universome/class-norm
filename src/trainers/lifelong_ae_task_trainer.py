import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.trainers.task_trainer import TaskTrainer

class LifeLongAETaskTrainer(TaskTrainer):
    def _after_init_hook(self):
        if self.task_idx == 0:
            self.log_img_idx = random.sample(range(len(self.task_ds_test)), 10)
        else:
            self.log_img_idx = self.get_previous_trainer().log_img_idx

    def train_on_batch(self, batch):
        self.model.train()
        x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
        loss = F.mse_loss(self.model(x), x)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('loss/train', loss.item(), self.num_iters_done)

    def construct_optimizer(self):
        if self.task_idx == 0:
            return super(LifeLongAETaskTrainer, self).construct_optimizer()
        else:
            return self.get_previous_trainer().optim

    def validate_ae(self, dataloader):
        all_losses = []
        self.model.eval()

        with torch.no_grad():
            #for x, _ in tqdm(dataloader, desc='Validating', total=len(dataloader)):
            for batch in dataloader:
                x = torch.from_numpy(np.array(batch[0])).to(self.device_name)
                losses = F.mse_loss(self.model(x), x, reduction='none')
                all_losses.extend(losses.cpu().tolist())

        return np.mean(all_losses)

    def on_epoch_done(self):
        if self.num_epochs_done % 10 == 0:
            self.writer.add_scalar(f'loss/train_full', self.validate_ae(self.train_dataloader), self.num_iters_done)
            self.writer.add_scalar(f'loss/val', self.validate_ae(self.test_dataloader), self.num_iters_done)

    def _after_train_hook(self):
        for task_idx, (_, test_ds) in enumerate(self.main_trainer.data_splits):
            dataloader = self.create_dataloader(test_ds, shuffle=False)
            loss = self.validate_ae(dataloader)
            self.writer.add_scalar(f'loss/val_for_task', loss, task_idx)
            self.log_images(test_ds, task_idx)

    def log_images(self, dataset, task_idx: int):
        self.model.eval()

        with torch.no_grad():
            imgs = torch.tensor([dataset[i][0] for i in self.log_img_idx])
            recs = self.model(imgs.to(self.device_name)).cpu()

            imgs = imgs.permute(0, 2, 3, 1).numpy()
            recs = recs.permute(0, 2, 3, 1).numpy()

        fig = plt.figure(figsize=(20, 5))
        plt.subplot(211)
        plt.title('[val] Ground Truth')
        plt.imshow(np.stack(imgs, axis=1).reshape(32, -1, 3), interpolation='nearest')

        plt.subplot(212)
        plt.title('[val] Reconstructions')
        plt.imshow(np.stack(recs, axis=1).reshape(32, -1, 3), interpolation='nearest')

        self.writer.add_figure(f'reconstructions', fig, task_idx)
