"""Run a batch of experiments, either in sequence on a local machine, or in parallel on a SLURM cluster.

The `--batch_config` command will point to a set of hyperparemeters, which are specified in a JSON file in the `configs`
directory.
"""
import argparse
from collections import namedtuple
import copy
from itertools import product
import json
import os
from pdb import set_trace as TT
import re
import traceback
import yaml

from config import BatchConfig
from cross_eval import vis_cross_eval
from main import main_experiment
from mazes import Mazes, main_mazes  # Weirdly need this to be able to load dataset of mazes.
# from cross_eval import vis_cross_eval


def submit_slurm_job(sbatch_file, experiment_name, job_time, job_cpus, job_gpus, job_mem):
    cmd = f"python main.py --load_config {experiment_name}"

    with open(sbatch_file) as f:
        content = f.read()
        job_name = 'nca_'
        # if args.evaluate:
            # job_name += 'eval_'
        job_name += str(experiment_name)
        content = re.sub(r'nca_(eval_)?.+', job_name, content)
        ##SBATCH --gres=gpu:1
        gpu_str = f"#SBATCH --gres=gpu:{job_gpus}" if job_gpus > 0 else f"##SBATCH --gres=gpu:1"
        content = re.sub(r'#+SBATCH --gres=gpu:\d+', gpu_str, content)
        content = re.sub(r'#SBATCH --time=\d+:', '#SBATCH --time={}:'.format(job_time), content)
        content = re.sub(r'#SBATCH --cpus-per-task=\d+', '#SBATCH --cpus-per-task={}'.format(job_cpus), content)
        content = re.sub(r'#SBATCH --mem=\d+GB', '#SBATCH --mem={}GB'.format(job_mem), content)
        cmd = '\n' + cmd
        new_content = re.sub('\n.*python main.py.*', cmd, content)

    with open(sbatch_file, 'w') as f:
        f.write(new_content)

    os.system('sbatch {}'.format(sbatch_file))


def dump_config(exp_name, exp_config):
    with open(os.path.join('configs', 'auto', f'{exp_name}.json'), 'w') as f:
        json.dump(exp_config, f, indent=4)


def main_batch():
    batch_cfg = BatchConfig()
    gen_new_data = batch_cfg.gen_new_data
    job_time = 48
    
    with open(os.path.join('configs', f'{batch_cfg.batch_hyperparams}.yaml')) as f:
        batch_hyperparams = yaml.safe_load(f)

    # Generate dataset of mazes if necessary.
    if gen_new_data:
        if 'n_data' in batch_hyperparams:
            setattr(batch_cfg, 'n_data', max(batch_hyperparams['n_data']))
        main_mazes(batch_cfg)


    if batch_cfg.load_all:
        # Create an experiment config for each log folder present in the `runs` directory.
        overwrite_args = set({'load', 'evaluate', 'render'})
        exp_dirs = []
        exp_configs = []
        for (dirpath, dirnames, filenames) in os.walk('runs'):
            exp_dirs.extend(dirnames)
            break
        for exp_dir in exp_dirs:
            exp_config = copy.deepcopy(batch_cfg)
            cfg_path = os.path.join('runs', exp_dir, 'config.json')
            if not os.path.exists(cfg_path):
                print(f"Experiment config path does not exist:\n{cfg_path}")
                print("Skipping experiment.")
                continue
            load_config = json.load(open(cfg_path))
            [setattr(exp_config, k, v) for k, v in load_config.items() if k not in overwrite_args]

            if batch_cfg.filter_by_config:
                # Exclude experiments that would not have been launched by the batch config.
                invalid_cfg = False
                for k, v in vars(exp_config).items():
                    if k in batch_hyperparams and v not in batch_hyperparams[k]:
                        invalid_cfg = True
                        break
                if invalid_cfg:
                    continue

            exp_configs.append(exp_config)

    else:
        # Create an experiment config for each combination of hyperparameters.
        exp_configs = [copy.deepcopy(batch_cfg)]
        for key, hyperparams in batch_hyperparams.items():
            old_exp_configs = exp_configs
            new_exp_configs = []
            for hyperparam in hyperparams:
                for exp_config in old_exp_configs:
                    exp_config = copy.deepcopy(exp_config)
                    setattr(exp_config, key, hyperparam)
                    new_exp_configs.append(exp_config)
            exp_configs = new_exp_configs
    
    # Validate experiment configs, setting unique experiment names and filtering out invalid configs (flagged by 
    # assertion errors in `config._validate()`).
    filtered_exp_configs = []
    for ec in exp_configs:
        try:
            # TODO: remove this once we implement `full_exp_name` inside `config.py`.
            if not batch_cfg.load_all:
                ec.set_exp_name()
            else:
                ec.validate()
            filtered_exp_configs.append(ec)
        except AssertionError as e:
            print("Experiment config is invalid:", e)
            print("Skipping experiment.")
    exp_configs = filtered_exp_configs

    if batch_cfg.vis_cross_eval:
        return vis_cross_eval(exp_configs, name=batch_cfg.batch_hyperparams)

    experiment_names = []
    for exp_cfg in exp_configs:
        if not batch_cfg.slurm:
            try:
                main_experiment(exp_cfg)
            except Exception as e:
                print("Experiment failed, with error:")
                print(traceback.format_exc())
                print("Skipping experiment.")
                continue

        else:
            raise NotImplementedError
        # elif not batch_cfg.load:
        #     experiment_names.append(exp_cfg.exp_name)
        #     dump_config(exp_cfg.exp_name, exp_cfg)
        # else:

        #     # Remove this eventually. Very ad hoc backward compatibility with broken experiment naming schemes:
        #     found_save_dir = False
        #     sd_i = 0
        #     if not os.path.isdir(exp_cfg.exp_name):
        #         print(f'No directory found for experiment at {exp_cfg.exp_name}.')
        #     else:
        #         exp_config['experiment_name'] = exp_cfg.exp_name
        #         experiment_names.append(exp_cfg.exp_name)
        #         dump_config(exp_cfg.exp_name, exp_config)
        #         break
        #     sd_i += 1
        #     if not found_save_dir:
        #         print('No save directory found for experiment. Skipping.')
        #     else:
        #         print('Found save dir: ', exp_cfg.exp_name)

    if batch_cfg.slurm:
        sbatch_file = os.path.join('slurm', 'run.sh')
        for experiment_name in experiment_names:
            # Because of our parallel evo/train implementation, we need an additional CPU for the remote trainer, and 
            # anoter for the local worker (actually the latter is not true, but... for breathing room).
            submit_slurm_job(sbatch_file=sbatch_file, experiment_name=experiment_name, job_time=job_time, job_cpus=1, \
                job_gpus=1, job_mem=16)


if __name__ == '__main__':
    main_batch()
