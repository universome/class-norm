import os
import click
import shutil
from distutils.dir_util import copy_tree
from typing import Tuple, List, Dict
import subprocess

from firelab.config import Config
from firelab.hpo import compute_hpo_vals_idx


def sbatch_args_to_str(sbatch_args: Dict) -> str:
    return ' '.join([f'--{k} {v}' for k, v in sbatch_args.items()])


BASE_PROJECT_DIR = '/home/skoroki/zslll'
DIRECTORIES_TO_COPY = [os.path.join(BASE_PROJECT_DIR, d) for d in ['slurm', 'configs', 'src']]
SBATCH_ARGS = {
    'time': '3:59:59',
    'gres': 'gpu:1',
    'cpus-per-task': '5',
    'mem': '64G'
}
SBATCH_ARGS_STR = sbatch_args_to_str(SBATCH_ARGS)


def generate_experiments_from_hpo_grid(hpo_grid):
    experiments_vals_idx = compute_hpo_vals_idx(hpo_grid)
    experiments_vals = [{p: hpo_grid[p][i] for p, i in zip(hpo_grid.keys(), idx)} for idx in experiments_vals_idx]

    return experiments_vals


def get_dataset_paths(args) -> Tuple[str, str]:
    if args.dataset.startswith('lsun_'):
        split = 'val' if args.debug else 'train'
        category_name = args.dataset[args.dataset.find('_') + 1:]
        source_data_dir = f'/ibex/scratch/skoroki/datasets/lsun/{category_name}_{split}_lmdb'
        target_data_dir = f'/tmp/skoroki/datasets/lsun/{category_name}_{split}_lmdb'
    elif args.dataset == 'ffhq_thumbs':
        ds_debug_prefix = '-mini' if args.debug else ''
        source_data_dir = f'/ibex/scratch/skoroki/datasets/ffhq/thumbnails128x128{ds_debug_prefix}'
        target_data_dir = f'/tmp/skoroki/datasets/ffhq/thumbnails128x128{ds_debug_prefix}'
    elif args.dataset == 'celeba_thumbs':
        source_data_dir = '/ibex/scratch/skoroki/datasets/celeba/thumbnails128x128'
        target_data_dir = '/tmp/skoroki/datasets/celeba/thumbnails128x128'
    else:
        raise NotImplementedError

    return source_data_dir, target_data_dir


def copy_dirs(target_dir: os.PathLike, dirs_to_copy: List[os.PathLike]):
    for d in dirs_to_copy:
        target_d = os.path.join(target_dir, os.path.basename(d))
        copy_tree(d, target_d)


def convert_config_to_cli_args(config: Config) -> str:
    conf_dict = {p.replace('|', '.'): v for p, v in config.to_dict().items()}
    cli_args = ' '.join([f'--config.{p} {v}' for p, v in conf_dict.items()])

    return cli_args


def create_project_dir(project_dir: os.PathLike, force_delete: bool=False) -> bool:
    if os.path.exists(project_dir):
        if force_delete or click.confirm(f'Dir {project_dir} already exists. Remove it?', default=False):
            shutil.rmtree(project_dir)
        else:
            print('User refused to delete an existing project dir.')
            return False

    os.makedirs(project_dir)
    copy_dirs(project_dir, DIRECTORIES_TO_COPY)

    if are_there_uncommitted_changes():
        print('Warning: there are uncommited changes!')

    print(f'Created a project dir: {project_dir}')

    return True


def get_git_hash() -> str:
    return subprocess \
        .check_output(['git', 'rev-parse', '--short', 'HEAD']) \
        .decode("utf-8") \
        .strip()


def are_there_uncommitted_changes() -> bool:
    return len(subprocess.check_output('git status -s'.split()).decode("utf-8")) > 0
