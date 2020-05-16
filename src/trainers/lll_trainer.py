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

from src.dataloaders.load_data import load_data
from src.utils.data_utils import split_classes_for_tasks, get_train_test_data_splits
from src.utils.constants import DEBUG

from src.trainers.basic_task_trainer import BasicTaskTrainer
from src.trainers.agem_task_trainer import AgemTaskTrainer
from src.trainers.ewc_task_trainer import EWCTaskTrainer
from src.trainers.mas_task_trainer import MASTaskTrainer
from src.trainers.joint_task_trainer import JointTaskTrainer
from src.trainers.dem_task_trainer import DEMTaskTrainer
from src.trainers.icarl_task_trainer import iCarlTaskTrainer
from src.trainers.multi_proto_task_trainer import MultiProtoTaskTrainer

from src.utils.data_utils import construct_output_mask, compute_class_centroids, flatten
from src.utils.training_utils import normalize
from src.dataloaders.utils import create_custom_dataset, extract_features_for_dataset
from src.utils.metrics import compute_unseen_classes_acc_history, compute_seen_classes_acc_history, compute_individual_accs_matrix, compute_forgetting_measure

TASK_TRAINERS = {
    'basic': BasicTaskTrainer,
    'agem': AgemTaskTrainer,
    'ewc': EWCTaskTrainer,
    'mas': MASTaskTrainer,
    'joint': JointTaskTrainer,
    'dem': DEMTaskTrainer,
    'icarl': iCarlTaskTrainer,
    'multi_proto': MultiProtoTaskTrainer
}

MODELS = {
    'simple_classifier': ResnetClassifier,
    'feat_classifier': FeatClassifier,
}

class LLLTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super(LLLTrainer, self).__init__(config)

        self.logger.info(f'Implementation method: {self.config.task_trainer}')
        self.logger.info(f'Using device: {self.device_name}')

        self.episodic_memory = []
        self.episodic_memory_output_mask = []
        self.logits_history = []
        self.train_logits_history = []
        self.knn_logits_history = []
        self.golden_logits_history = []

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
            self.config.data, self.config.hp.get('img_target_shape'))
        self.class_splits = split_classes_for_tasks(self.config.lll_setup, self.config.random_seed)
        classes_used = set(flatten(self.class_splits))

        if len(classes_used) < self.config.data.num_classes:
            self.ds_train = self.ds_train.filter_out_classes(classes_used)
            self.ds_test = self.ds_test.filter_out_classes(classes_used)

        self.data_splits = get_train_test_data_splits(self.class_splits, self.ds_train, self.ds_test)

        for task_idx, task_classes in enumerate(self.class_splits):
            print(f'[Task {task_idx}]:', task_classes)

    def start(self):
        self.init()
        self.num_tasks_learnt = 0
        self.task_trainers = [] # TODO: this is memory-leaky :|

        for task_idx in range(self.config.lll_setup.num_tasks):
            # print(f'Starting task #{task_idx}')

            self.save_logits_history()

            task_trainer = TASK_TRAINERS[self.config.task_trainer](self, task_idx)

            self.task_trainers.append(task_trainer)

            if self.config.has('start_task') and self.num_tasks_learnt < self.config.start_task:
                self.num_tasks_learnt += 1
                continue
            else:
                task_trainer.start()

            self.num_tasks_learnt += 1

            if self.config.get('logging.print_accuracy_after_task'):
                print(f'Train accuracy: {task_trainer.compute_train_accuracy()}')
                print(f'Test accuracy: {task_trainer.compute_test_accuracy()}')

            if self.config.task_trainer == 'agem':
                task_trainer.update_episodic_memory()

            if self.config.get('should_checkpoint', False):
                self.task_checkpoint(task_idx)

        self.save_logits_history()
        self.save_experiment_data()

        if self.config.get('logging.print_unseen_accuracy'):
            values = self.compute_unseen_accuracy()
            print(f'Unseen accuracy (mean: {np.mean(values): .03f}): {", ".join([f"{a: 0.4f}" for a in values])}')

        if self.config.get('logging.print_forgetting'):
            values = self.compute_forgetting()
            print(f'Forgetting (mean: {np.mean(values): .03f}): {", ".join([f"{a: 0.4f}" for a in values])}')


    def save_logits_history(self):
        if self.config.get('logging.save_logits'):
            self.logits_history.append(self.run_inference(self.ds_test))

        if DEBUG: return
        # if self.config.get('logging.save_knn_logits'):
        #     self.knn_logits_history.append(self.run_inference(self.ds_test, model_kwargs={"aggregation_type": "shortest_distance"}))

        # if self.config.get('logging.save_golden_logits'):
        #     self.golden_logits_history.append(self.run_inference(self.ds_test, model_kwargs={"aggregation_type": "golden_prototype"}))

        if self.config.get('logging.save_train_logits'):
            self.train_logits_history.append(self.run_inference(self.ds_train))

    def run_inference(self, dataset: List[Tuple[np.ndarray, int]], model_kwargs={}):
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=self.config.get('inference_batch_size', self.config.hp.batch_size), num_workers=4)

        with torch.no_grad():
            if self.config.hp.get('use_oracle_prototypes') or self.config.hp.get('use_oracle_softmax_mean'):
                ds_train_feats = extract_features_for_dataset(self.ds_train, self.model.embedder, self.device_name, 256)
                feats = extract_features_for_dataset(dataset, self.model.embedder, self.device_name, 256)
                feats = normalize(torch.from_numpy(np.array([x for x, _ in feats])), self.config.hp.head.scale.value) # [ds_size, hid_dim]

                if self.config.hp.get('use_oracle_prototypes'):
                    prototypes_raw = compute_class_centroids(ds_train_feats, self.config.data.num_classes) # [num_classes, hid_dim]
                    prototypes = normalize(torch.from_numpy(prototypes_raw).float(), self.config.hp.head.scale.value) # [num_classes, hid_dim]

                    # Logits is the dot-product with the prototypes
                    logits = (feats @ prototypes.t()).cpu().numpy() # [ds_size, num_classes]
                else:
                    max_num_protos_per_class = 25
                    ds_size = len(dataset)
                    n_classes = self.config.data.num_classes

                    feats_train = normalize(torch.from_numpy(np.array([x for x, _ in ds_train_feats])), self.config.hp.head.scale.value) # [train_ds_size, hid_dim]
                    classes_train = self.ds_train.labels
                    class_idx = [np.where(classes_train == c)[0][:max_num_protos_per_class] for c in range(n_classes)] # [n_classes, n_protos]
                    feats_train = torch.stack([feats_train[idx] for idx in class_idx]) # [n_classes, n_protos, hid_dim]
                    logits_mp = feats_train @ feats.t() # [n_classes, n_protos, ds_size]
                    logits_mp = logits_mp.permute(2, 0, 1).view(ds_size, -1) # [ds_size, n_classes * n_protos]
                    probs_mp = logits_mp.softmax(dim=1)
                    logits = probs_mp.view(ds_size, n_classes, max_num_protos_per_class).sum(dim=2).log() # [ds_size, n_classes]
                    logits = logits.cpu().numpy()
            else:
                logits = [self.model(torch.from_numpy(np.array(b)).to(self.device_name), **model_kwargs).cpu().numpy() for b, _ in dataloader]
                logits = np.vstack(logits)

        return logits

    def save_experiment_data(self):
        np.save(os.path.join(self.paths.custom_data_path, 'logits_history'), self.logits_history)
        np.save(os.path.join(self.paths.custom_data_path, 'train_logits_history'), self.train_logits_history)
        np.save(os.path.join(self.paths.custom_data_path, 'knn_logits_history'), self.knn_logits_history)
        np.save(os.path.join(self.paths.custom_data_path, 'golden_logits_history'), self.golden_logits_history)
        np.save(os.path.join(self.paths.custom_data_path, 'class_splits'), self.class_splits)
        np.save(os.path.join(self.paths.custom_data_path, 'targets'), self.ds_test.labels)
        np.save(os.path.join(self.paths.custom_data_path, 'train_targets'), self.ds_train.labels)

        if self.config.get('logging.save_final_model'):
            self.checkpoint('final-model')

    def task_checkpoint(self, curr_task_idx: int):
        self.checkpoint(f'model-task-{curr_task_idx}')

    def checkpoint(self, model_name: str):
        path = os.path.join(self.paths.checkpoints_path, f'{model_name}.pt')
        torch.save(self.model.state_dict(), path)

    def compute_forgetting(self):
        """Computes forgetting for the latest task"""
        targets = self.ds_test.labels
        accs_matrix = compute_individual_accs_matrix(self.logits_history[1:], targets, self.class_splits)

        return [compute_forgetting_measure(accs_matrix, i) for i in range(1, len(accs_matrix))]

    def compute_unseen_accuracy(self):
        """Computes unseen accuracy for the latest task"""
        targets = self.ds_test.labels

        return compute_unseen_classes_acc_history(self.logits_history[:-1], targets, self.class_splits, restrict_space=False)

    def compute_harmonic_mean_accuracy(self) -> np.ndarray:
        targets = self.ds_test.labels
        values_unseen = compute_unseen_classes_acc_history(self.logits_history[:-1], targets, self.class_splits, restrict_space=False)
        values_seen = compute_seen_classes_acc_history(self.logits_history[1:], targets, self.class_splits, restrict_space=False)
        values_unseen = np.array(values_unseen)
        values_seen = np.array(values_seen)
        values = 2 * values_unseen[1:] * values_seen[:-1] / (values_unseen[1:] + values_seen[:-1] + 1e-8)

        return values
