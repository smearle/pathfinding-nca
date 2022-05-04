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
import yaml

from config import BatchConfig, get_exp_name
from main import experiment_main
from mazes import Mazes  # Weirdly need this to be able to load dataset of mazes.
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


def batch_main():
    cfg = BatchConfig()
    job_time = 48
    
    with open(os.path.join('configs', 'batch.yaml')) as f:
        batch_hyperparams = yaml.safe_load(f)

    exp_configs = [copy.deepcopy(cfg)]

    # Create an experiment config for each combination of hyperparameters.
    for key, hyperparams in batch_hyperparams.items():
        old_exp_configs = exp_configs
        new_exp_configs = []
        for hyperparam in hyperparams:
            for exp_config in old_exp_configs:
                print(exp_config)
                exp_config = copy.deepcopy(exp_config)
                setattr(exp_config, key, hyperparam)
                new_exp_configs.append(exp_config)
        exp_configs = new_exp_configs

    experiment_names = []
    for exp_cfg in exp_configs:
        if not cfg.slurm:
            experiment_main(exp_cfg)

        elif not cfg.load:
            experiment_name = get_exp_name(cfg)
            experiment_names.append(experiment_name)
            dump_config(experiment_name, exp_cfg)
        else:

            # Remove this eventually. Very ad hoc backward compatibility with broken experiment naming schemes:
            found_save_dir = False
            sd_i = 0
            experiment_name = get_exp_name(exp_cfg)
            if not os.path.isdir(experiment_name):
                print(f'No directory found for experiment at {experiment_name}.')
            else:
                exp_config['experiment_name'] = experiment_name
                experiment_names.append(experiment_name)
                dump_config(experiment_name, exp_config)
                break
            sd_i += 1
            if not found_save_dir:
                print('No save directory found for experiment. Skipping.')
            else:
                print('Found save dir: ', experiment_name)

    if cfg.vis_cross_eval:
        # vis_cross_eval(exp_configs)
        raise NotImplementedError

        return 

    if cfg.slurm:
        sbatch_file = os.path.join('slurm', 'run.sh')
        for experiment_name in experiment_names:
            # Because of our parallel evo/train implementation, we need an additional CPU for the remote trainer, and 
            # anoter for the local worker (actually the latter is not true, but... for breathing room).
            submit_slurm_job(sbatch_file=sbatch_file, experiment_name=experiment_name, job_time=job_time, job_cpus=1, \
                job_gpus=1, job_mem=16)



if __name__ == '__main__':
    batch_main()
