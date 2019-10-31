import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from tqdm import tqdm

from src.models.classifier import ZSClassifier, ResnetClassifier
from src.models.feat_gan_classifier import FeatGANClassifier
from src.dataloaders import cub, awa
from src.utils.data_utils import split_classes_for_tasks, get_train_test_data_splits
from src.trainers.basic_task_trainer import BasicTaskTrainer
from src.trainers.agem_task_trainer import AgemTaskTrainer
from src.trainers.ewc_task_trainer import EWCTaskTrainer
from src.trainers.mas_task_trainer import MASTaskTrainer
from src.trainers.mergazsl_task_trainer import MeRGAZSLTaskTrainer
from src.utils.data_utils import construct_output_mask
from src.dataloaders.utils import extract_resnet18_features_for_dataset
from src.utils.metrics import (
    compute_average_accuracy,
    compute_forgetting_measure,
    compute_learning_curve_area,
    compute_ausuc
)

class LLLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super(LLLTrainer, self).__init__(config)

        self.logger.info(f'Implementation method: {self.config.task_trainer}')
        self.logger.info(f'Using device: {self.device_name}')

        self.episodic_memory = []
        self.episodic_memory_output_mask = []
        self.zst_accs = []
        self.accs_history = []
        self.logits_history = []
        self.ausuc_scores = []
        self.ausuc_accs = []

    def init_models(self):
        if self.config.hp.get('use_class_attrs'):
            if self.config.hp.get('model_type', 'simple_classifier') == 'simple_classifier':
                self.model = ZSClassifier(self.class_attributes, pretrained=self.config.hp.pretrained)
            elif self.config.hp.model_type == 'feat_gan_classifier':
                self.model = FeatGANClassifier(self.class_attributes, self.config.hp.model_config)
            else:
                raise NotImplementedError(f'Unknown model type {self.config.hp.model_type}')
        else:
            assert self.config.hp.get('model_type', 'simple_classifier') == 'simple_classifier'
            self.model = ResnetClassifier(self.config.data.num_classes, pretrained=self.config.hp.pretrained)

        self.model = self.model.to(self.device_name)

    def init_optimizers(self):
        if self.config.hp.get('model_type', 'simple_classifier') == 'simple_classifier':
            # TODO: without momentum?!
            self.optim = torch.optim.SGD(self.model.parameters(), **self.config.hp.optim_kwargs.to_dict())
        elif self.config.hp.model_type == 'feat_gan_classifier':
            # TODO: well, now this does not look like a good code...
            self.optim = {
                'gen': torch.optim.Adam(self.model.generator.parameters(), **self.config.hp.model_config.gen_optim_kwargs.to_dict()),
                'discr': torch.optim.Adam(self.model.discriminator.parameters(), **self.config.hp.model_config.discr_optim_kwargs.to_dict()),
            }
        else:
            raise NotImplementedError(f'Unknown model type {self.config.hp.model_type}')

    def init_dataloaders(self):
        if self.config.data.name == 'CUB':
            self.ds_train = cub.load_dataset(self.config.data.dir, is_train=True, target_shape=self.config.hp.img_target_shape)
            self.ds_test = cub.load_dataset(self.config.data.dir, is_train=False, target_shape=self.config.hp.img_target_shape)
            self.class_attributes = cub.load_class_attributes(self.config.data.dir)
        elif self.config.data.name == 'AWA':
            self.ds_train = awa.load_dataset(self.config.data.dir, split='train', target_shape=self.config.hp.img_target_shape)
            self.ds_test = awa.load_dataset(self.config.data.dir, split='test', target_shape=self.config.hp.img_target_shape)
            self.class_attributes = awa.load_class_attributes(self.config.data.dir)
        else:
            raise NotImplementedError(f'Unkown dataset: {self.config.data.name}')

        if self.config.hp.get('embed_data'):
            self.ds_train = extract_resnet18_features_for_dataset(self.ds_train)
            self.ds_test = extract_resnet18_features_for_dataset(self.ds_test)

        self.class_splits = split_classes_for_tasks(
            self.config.data.num_classes, self.config.data.num_tasks,
            self.config.data.num_classes_per_task, self.config.data.get('num_reserved_classes', 0))
        self.data_splits = get_train_test_data_splits(self.class_splits, self.ds_train, self.ds_test)

        for task_idx, split in enumerate(self.class_splits):
            print(f'[Task {task_idx}]:', self.class_splits[task_idx].tolist())

    def start(self):
        self.init()
        self.num_tasks_learnt = 0
        self.task_trainers = [] # TODO: this is memory-leaky :|

        for task_idx in range(self.config.data.num_tasks):
            print(f'Starting task #{task_idx}', end='')

            task_trainer = self.construct_trainer(task_idx)

            if self.config.get('logging.save_logits'):
                self.logits_history.append(self.run_inference(self.ds_test))

            if self.config.get('metrics.ausuc'):
                self.track_ausuc()

            self.task_trainers.append(task_trainer)
            self.zst_accs.append(task_trainer.compute_test_accuracy())
            print(f'. ZST accuracy: {self.zst_accs[-1]}')
            task_trainer.start()
            self.num_tasks_learnt += 1
            print(f'Train accuracy: {task_trainer.compute_train_accuracy()}')
            print(f'Test accuracy: {task_trainer.compute_test_accuracy()}')

            if self.config.task_trainer == 'agem':
                task_trainer.extend_episodic_memory()

            if self.config.get('metrics.average_accuracy') or self.config.get('metrics.forgetting_measure'):
                self.validate()

        self.compute_metrics()
        self.save_experiment_data()

    def compute_metrics(self):
        if self.config.get('metrics.average_accuracy'):
            print('Average Accuracy:', compute_average_accuracy(self.accs_history))

        if self.config.get('metrics.forgetting_measure'):
            print('Forgetting Measure:', compute_forgetting_measure(self.accs_history))

        if self.config.get('metrics.lca_num_batches', -1) >= 0:
            lca_accs = [t.test_acc_batch_history for t in self.task_trainers]
            lca_n_batches = min(self.config.metrics.lca_num_batches, min([len(accs) - 1 for accs in lca_accs]))
            print(f'Learning Curve Area [beta = {lca_n_batches}]:', compute_learning_curve_area(lca_accs, lca_n_batches))

        if self.config.get('metrics.ausuc'):
            self.track_ausuc()
            print('Mean ausuc:', np.mean(self.ausuc_scores))

        if self.config.get('logging.save_logits'):
            self.logits_history.append(self.run_inference(self.ds_test))

    def track_ausuc(self):
        logits = self.run_inference(self.ds_test)
        targets = np.array([y for _, y in self.ds_test])
        seen_classes = np.unique(self.class_splits[:self.num_tasks_learnt])
        seen_classes_mask = construct_output_mask(seen_classes, self.config.data.num_classes)
        ausuc, ausuc_accs = compute_ausuc(logits, targets, seen_classes_mask)

        self.ausuc_scores.append(ausuc)
        self.ausuc_accs.append(ausuc_accs)

        self.writer.add_scalar('AUSUC', ausuc, self.num_tasks_learnt)

    def run_inference(self, dataset: List[Tuple[np.ndarray, int]]):
        self.model.eval()

        examples = [x for x, _ in dataset]
        dataloader = DataLoader(examples, batch_size=self.config.get('inference_batch_size', self.config.hp.batch_size))

        with torch.no_grad():
            logits = [self.model(torch.tensor(b).to(self.device_name)).cpu().numpy() for b in dataloader]
            logits = np.vstack(logits)

        return logits

    def save_experiment_data(self):
        np.save(os.path.join(self.paths.custom_data_path, 'zst_accs'), self.zst_accs)
        np.save(os.path.join(self.paths.custom_data_path, 'accs_history'), self.accs_history)
        np.save(os.path.join(self.paths.custom_data_path, 'test_acc_batch_histories'),
            [t.test_acc_batch_history for t in self.task_trainers])
        np.save(os.path.join(self.paths.custom_data_path, 'ausuc_scores'), self.ausuc_scores)
        np.save(os.path.join(self.paths.custom_data_path, 'ausuc_accs'), self.ausuc_accs)
        np.save(os.path.join(self.paths.custom_data_path, 'logits_history'), self.logits_history)
        np.save(os.path.join(self.paths.custom_data_path, 'class_splits'), self.class_splits)

    def construct_trainer(self, task_idx: int) -> "TaskTrainer":
        if self.config.task_trainer == 'basic':
            return BasicTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'agem':
            return AgemTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'ewc':
            return EWCTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'mas':
            return MASTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'mergazsl':
            return MeRGAZSLTaskTrainer(self, task_idx)
        else:
            raise NotImplementedError(f'Unknown task trainer: {self.config.task_trainer}')

    def compute_test_accs(self) -> List[float]:
        """Computes model test accuracy on all the tasks"""
        accuracies = []

        for task_idx in tqdm(range(self.config.data.num_tasks), desc='[Validating]'):
            trainer = self.construct_trainer(task_idx)
            accuracy = trainer.compute_test_accuracy()
            accuracies.append(accuracy)

        return accuracies

    def validate(self):
        accs = self.compute_test_accs()

        for task_id, acc in enumerate(accs):
            self.writer.add_scalar('Task_test_acc/{}', acc, self.num_tasks_learnt)

        self.accs_history.append(accs)
