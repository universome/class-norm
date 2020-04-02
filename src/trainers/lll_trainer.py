import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from tqdm import tqdm
import yaml

from src.models.classifier import ResnetClassifier, FeatClassifier
from src.models.feat_gan_classifier import FeatGANClassifier
from src.models.gan import GAN
from src.models.gan_64x64 import GAN64x64
from src.models.lgm import LGM
from src.models.lgm_vae import LGMVAE
from src.models.autoencoder import AutoEncoder

from src.dataloaders.load_data import load_data
from src.dataloaders.utils import imagenet_normalization
from src.utils.data_utils import split_classes_for_tasks, get_train_test_data_splits

from src.trainers.basic_task_trainer import BasicTaskTrainer
from src.trainers.agem_task_trainer import AgemTaskTrainer
from src.trainers.ewc_task_trainer import EWCTaskTrainer
from src.trainers.mas_task_trainer import MASTaskTrainer
from src.trainers.mergazsl_task_trainer import MeRGAZSLTaskTrainer
from src.trainers.joint_task_trainer import JointTaskTrainer
from src.trainers.gm_gan_task_trainer import GMGANTaskTrainer
from src.trainers.lgm_task_trainer import LGMTaskTrainer
from src.trainers.lgm_vae_task_trainer import LGMVAETaskTrainer
from src.trainers.lgm_aegan_task_trainer import LGMAEGANTaskTrainer
from src.trainers.lifelong_ae_task_trainer import LifeLongAETaskTrainer
from src.trainers.dem_task_trainer import DEMTaskTrainer
from src.trainers.icarl_task_trainer import iCarlTaskTrainer
from src.trainers.multi_proto_task_trainer import MultiProtoTaskTrainer

from src.utils.data_utils import construct_output_mask, filter_out_classes
from src.dataloaders.utils import create_custom_dataset


TASK_TRAINERS = {
    'basic': BasicTaskTrainer,
    'agem': AgemTaskTrainer,
    'ewc': EWCTaskTrainer,
    'mas': MASTaskTrainer,
    'mergazsl': MeRGAZSLTaskTrainer,
    'joint': JointTaskTrainer,
    'gm_gan': GMGANTaskTrainer,
    'lgm': LGMTaskTrainer,
    'lgm_vae': LGMVAETaskTrainer,
    'lgm_aegan': LGMAEGANTaskTrainer,
    'lifelong_ae': LifeLongAETaskTrainer,
    'dem': DEMTaskTrainer,
    'icarl': iCarlTaskTrainer,
    'multi_proto': MultiProtoTaskTrainer
}

MODELS = {
    'simple_classifier': ResnetClassifier,
    'feat_gan_classifier': FeatGANClassifier,
    'lgm': LGM,
    'lgm_vae': LGMVAE,
    'feat_classifier': FeatClassifier,
    'gm_gan': GAN,
    'gm_gan_64x64': GAN64x64,
    'autoencoder': AutoEncoder
}

class LLLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super(LLLTrainer, self).__init__(config)

        self.logger.info(f'Implementation method: {self.config.task_trainer}')
        self.logger.info(f'Using device: {self.device_name}')

        self.episodic_memory = []
        self.episodic_memory_output_mask = []
        self.logits_history = []

        self.save_config()

    def save_config(self):
        config_yml = yaml.safe_dump(self.config.to_dict())
        config_yml = config_yml.replace('\n', '  \n') # Because tensorboard uses markdown
        self.writer.add_text('Config', config_yml, self.num_iters_done)

    def init_models(self):
        self.model = self.create_model()

    def create_model(self):
        print(f'Class attributes are switched {"on" if self.config.hp.get("use_class_attrs") else "off"}.')

        if self.config.hp.get('use_class_attrs'):
            model = MODELS[self.config.hp.model.type](self.config, self.class_attributes)
        else:
            model = MODELS[self.config.hp.model.type](self.config)

        if self.config.has('load_from_checkpoint'):
            model.load_state_dict(torch.load(self.config.load_from_checkpoint))

        return model.to(self.device_name)

    def init_dataloaders(self):
        self.ds_train, self.ds_test, self.class_attributes = load_data(
            self.config.data, self.config.hp.get('img_target_shape'), low_memory=self.config.get('low_memory', False))

        if self.config.data.has('classes_to_use'):
            self.ds_train = filter_out_classes(self.ds_train, self.config.data.classes_to_use)
            self.ds_test = filter_out_classes(self.ds_test, self.config.data.classes_to_use)

        self.class_splits = split_classes_for_tasks(self.config.lll_setup)
        self.data_splits = get_train_test_data_splits(self.class_splits, self.ds_train, self.ds_test)

        for task_idx, task_classes in enumerate(self.class_splits):
            print(f'[Task {task_idx}]:', task_classes)

    def start(self):
        self.init()
        self.num_tasks_learnt = 0
        self.task_trainers = [] # TODO: this is memory-leaky :|

        for task_idx in range(self.config.lll_setup.num_tasks):
            print(f'Starting task #{task_idx}')

            if self.config.get('logging.save_logits'):
               self.logits_history.append(self.run_inference(self.ds_test))

            task_trainer = TASK_TRAINERS[self.config.task_trainer](self, task_idx)

            self.task_trainers.append(task_trainer)

            if self.config.has('start_task') and self.num_tasks_learnt < self.config.start_task:
                pass
            else:
                task_trainer.start()

            self.num_tasks_learnt += 1

            if self.config.get('logging.print_accuracy_after_task'):
                print(f'Train accuracy: {task_trainer.compute_train_accuracy()}')
                print(f'Test accuracy: {task_trainer.compute_test_accuracy()}')

            if self.config.task_trainer == 'agem':
                task_trainer.update_episodic_memory()

            if self.config.get('should_checkpoint', False):
                self.checkpoint(task_idx)

        if self.config.get('logging.save_logits'):
            self.logits_history.append(self.run_inference(self.ds_test))

        self.save_experiment_data()

    def run_inference(self, dataset: List[Tuple[np.ndarray, int]]):
        self.model.eval()

        if self.config.get('low_memory'):
            dataset = create_custom_dataset(dataset, self.config.hp.img_target_shape)

        dataloader = DataLoader(dataset, batch_size=self.config.get('inference_batch_size', self.config.hp.batch_size))

        with torch.no_grad():
            logits = [self.model(torch.from_numpy(np.array(b)).to(self.device_name)).cpu().numpy() for b, _ in dataloader]
            logits = np.vstack(logits)

        return logits

    def save_experiment_data(self):
        np.save(os.path.join(self.paths.custom_data_path, 'logits_history'), self.logits_history)
        np.save(os.path.join(self.paths.custom_data_path, 'class_splits'), self.class_splits)
        np.save(os.path.join(self.paths.custom_data_path, 'targets'), [y for _, y in self.ds_test])

    def checkpoint(self, curr_task_idx: int):
        path = os.path.join(self.paths.checkpoints_path, f'model-task-{curr_task_idx}.pt')
        torch.save(self.model.state_dict(), path)
