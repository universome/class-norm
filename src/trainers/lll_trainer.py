import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from tqdm import tqdm
import yaml

from src.models.classifier import ZSClassifier, ResnetClassifier
from src.models.feat_gan_classifier import FeatGANClassifier
from src.models.feat_vae import FeatVAEClassifier
from src.models.gan import GAN
from src.models.gan_64x64 import GAN64x64
from src.models.lat_gm import LatGM
from src.models.lat_gm_vae import LatGMVAE

from src.dataloaders.load_data import load_data
from src.dataloaders.utils import imagenet_normalization
from src.utils.data_utils import split_classes_for_tasks, get_train_test_data_splits

from src.trainers.basic_task_trainer import BasicTaskTrainer
from src.trainers.agem_task_trainer import AgemTaskTrainer
from src.trainers.ewc_task_trainer import EWCTaskTrainer
from src.trainers.mas_task_trainer import MASTaskTrainer
from src.trainers.mergazsl_task_trainer import MeRGAZSLTaskTrainer
from src.trainers.joint_task_trainer import JointTaskTrainer
from src.trainers.genmem_gan_task_trainer import GenMemGANTaskTrainer
from src.trainers.lat_gm_task_trainer import LatGMTaskTrainer
from src.trainers.lat_gm_vae_task_trainer import LatGMVAETaskTrainer

from src.utils.data_utils import construct_output_mask, filter_out_classes

from src.utils.metrics import (
    compute_average_accuracy,
    compute_forgetting_measure,
    compute_learning_curve_area,
    compute_ausuc,
    compute_acc_for_classes,
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
        self.lca_accs = []
        self.iter_accs_history = []

        self.save_config()

    def save_config(self):
        config_yml = yaml.safe_dump(self.config.to_dict())
        config_yml = config_yml.replace('\n', '  \n') # Because tensorboard uses markdown
        self.writer.add_text('Config', config_yml, self.num_iters_done)

    def init_models(self):
        self.model = self.create_model()

    def create_model(self):
        if self.config.hp.get('use_class_attrs'):
            if self.config.hp.model.type == 'simple_classifier':
                model = ZSClassifier(self.class_attributes, pretrained=self.config.hp.pretrained)
            elif self.config.hp.model.type == 'feat_gan_classifier':
                model = FeatGANClassifier(self.config.hp.model, self.class_attributes)
            elif self.config.hp.model.type== 'feat_vae_classifier':
                model = FeatVAEClassifier(self.config.hp.model, self.class_attributes)
            elif self.config.hp.model.type == 'lat_gm':
                model = LatGM(self.config.hp.model, self.class_attributes)
            elif self.config.hp.model.type == 'lat_gm_vae':
                model = LatGMVAE(self.config.hp.model, self.class_attributes)
            else:
                raise NotImplementedError(f'Unknown model type {self.config.hp.model.type}')
        else:
            if self.config.hp.model.type == 'simple_classifier':
                model = ResnetClassifier(self.config.data.num_classes, pretrained=self.config.hp.pretrained)
            elif self.config.hp.model.type == 'genmem_gan':
                model = GAN(self.config.hp.model)
            elif self.config.hp.model.type == 'genmem_gan_64x64':
                model = GAN64x64(self.config.hp.model)
            else:
                raise NotImplementedError(f'Unkown model type to use without attrs: {self.config.hp.model.type}')

        if self.config.has('load_from_checkpoint'):
            model.load_state_dict(torch.load(self.config.load_from_checkpoint))

        return model.to(self.device_name)

    def init_optimizers(self):
        if self.config.hp.model.type == 'simple_classifier':
            # TODO: without momentum?!
            self.optim = torch.optim.Adam(self.model.parameters(), **self.config.hp.optim.kwargs.to_dict())
        elif self.config.hp.model.type in (
                'feat_gan_classifier',
                'feat_vae_classifier',
                'genmem_gan',
                'genmem_gan_64x64',
                'lat_gm',
                'lat_gm_vae'):
            self.optim = {} # We'll set this later in task trainer
        else:
            raise NotImplementedError(f'Unknown model type {self.config.hp.model.type}')

    def init_dataloaders(self):
        self.ds_train, self.ds_test, self.class_attributes = load_data(
            self.config.data, self.config.hp.get('img_target_shape'), self.config.hp.get('embed_data', False))

        if self.config.data.has('classes_to_use'):
            self.ds_train = filter_out_classes(self.ds_train, self.config.data.classes_to_use)
            self.ds_test = filter_out_classes(self.ds_test, self.config.data.classes_to_use)

        self.class_splits = split_classes_for_tasks(
            self.config.data.num_classes, self.config.data.num_tasks,
            self.config.data.num_classes_per_task, self.config.data.get('num_reserved_classes', 0))
        self.data_splits = get_train_test_data_splits(self.class_splits, self.ds_train, self.ds_test)

        for task_idx, task_classes in enumerate(self.class_splits):
            print(f'[Task {task_idx}]:', task_classes.tolist())

    def measure_task_trainer_lca(self, task_trainer: "TaskTrainer"):
        if self.config.get('metrics.lca_num_batches', -1) >= task_trainer.num_iters_done:
            assert len(self.lca_accs) >= task_trainer.task_idx

            if len(self.lca_accs) == task_trainer.task_idx: self.lca_accs.append([])

            self.lca_accs[task_trainer.task_idx].append(task_trainer.compute_test_accuracy())

    def measure_accuracy_after_iter(self, _):
        logits = self.run_inference(self.ds_test)
        targets = [y for _, y in self.ds_test]
        accs = [compute_acc_for_classes(logits, targets, cs) for cs in self.class_splits]
        self.iter_accs_history.append(accs)

    def start(self):
        self.init()
        self.num_tasks_learnt = 0
        self.task_trainers = [] # TODO: this is memory-leaky :|

        for task_idx in range(self.config.data.num_tasks):
            print(f'Starting task #{task_idx}')

            if self.config.get('logging.save_logits'):
               self.logits_history.append(self.run_inference(self.ds_test))

            if self.config.get('metrics.ausuc'):
                self.track_ausuc()

            task_trainer = self.construct_trainer(task_idx)

            if self.config.get('metrics.lca_num_batches', -1) >= 0:
                task_trainer.after_iter_done_callbacks.append(self.measure_task_trainer_lca)

            if self.config.get('logging.log_logits_after_each_iter', False):
                task_trainer.after_iter_done_callbacks.append(self.measure_accuracy_after_iter)

            self.task_trainers.append(task_trainer)

            if self.config.has('start_task') and self.num_tasks_learnt < self.config.start_task:
                pass
            else:
                task_trainer.start()

            self.num_tasks_learnt += 1
            print(f'Train accuracy: {task_trainer.compute_train_accuracy()}')
            print(f'Test accuracy: {task_trainer.compute_test_accuracy()}')

            if self.config.task_trainer == 'agem':
                task_trainer.extend_episodic_memory()

            if self.config.get('metrics.average_accuracy') or self.config.get('metrics.forgetting_measure'):
                self.validate()

            self.checkpoint(task_idx)

        self.compute_metrics()
        self.save_experiment_data()

    def compute_metrics(self):
        if self.config.get('metrics.average_accuracy'):
            print('Average Accuracy:', compute_average_accuracy(self.accs_history))

        if self.config.get('metrics.forgetting_measure'):
            print('Forgetting Measure:', compute_forgetting_measure(self.accs_history))

        if self.config.get('metrics.lca_num_batches', -1) >= 0:
            lca_n_batches = min(self.config.metrics.lca_num_batches, min([len(accs) - 1 for accs in self.lca_accs]))
            print(f'Learning Curve Area [beta = {lca_n_batches}]:', compute_learning_curve_area(self.lca_accs, lca_n_batches))

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
        np.save(os.path.join(self.paths.custom_data_path, 'targets'), [y for _, y in self.ds_test])
        np.save(os.path.join(self.paths.custom_data_path, 'iter_acc_history'), self.iter_accs_history)

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
        elif self.config.task_trainer == 'joint':
            return JointTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'genmem_gan':
            return GenMemGANTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'lat_gm':
            return LatGMTaskTrainer(self, task_idx)
        elif self.config.task_trainer == 'lat_gm_vae':
            return LatGMVAETaskTrainer(self, task_idx)
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

    def checkpoint(self, curr_task_idx: int):
        path = os.path.join(self.paths.checkpoints_path, f'model-task-{curr_task_idx}.pt')
        torch.save(self.model.state_dict(), path)
