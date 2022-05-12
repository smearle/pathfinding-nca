#@title Imports and Notebook Utilities
#%tensorflow_version 2.x

import json
import math
import os
from pdb import set_trace as TT
import shutil
import sys
from matplotlib import animation

import cv2
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch as th
import torchinfo
import wandb
import yaml

from configs.config import Config
from evaluate import evaluate
from mazes import load_dataset, Mazes, render_discrete
from models import BfsNCA, FixedBfsNCA, GCN, MLP, NCA
from models.nn import PathfindingNN
from render import render_trained
from train import train
from utils import Logger, VideoWriter, get_discrete_loss, get_mse_loss, to_path, load


np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


def main_experiment(cfg: Config=None):
    os.system('nvidia-smi -L')
    if th.cuda.is_available():
            print('Using GPU/CUDA.')
            th.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
            print('Not using GPU/CUDA, using CPU.')
        
    if cfg is None:
        cfg = Config()
        cfg.set_exp_name()

    model_cls = globals()[cfg.model]

    print(f"Running experiment with config:\n {OmegaConf.to_yaml(cfg)}")
    
    # Setup training
    model = model_cls(cfg)
    # Set a dummy initial maze state.
    model.reset(th.zeros(cfg.minibatch_size, cfg.n_in_chan, cfg.width + 2, cfg.width + 2), is_torchinfo_dummy=True)

    # if not (cfg.render or cfg.evaluate):
    #     dummy_input = model.seed(batch_size=cfg.minibatch_size)
    #     torchinfo.summary(model, input_size=dummy_input.shape)

    # param_n = sum(p.numel() for p in model.parameters())
    # print('model param count:', param_n)
    opt = th.optim.Adam(model.parameters(), cfg.learning_rate)
    try:
        maze_data_train, maze_data_val, maze_data_test = load_dataset(cfg)
    except FileNotFoundError as e:
        print("No maze data files found. Run `python mazes.py` to generate the dataset.")
        raise
    # maze_data_val.get_subset(cfg.n_val_data)
  # cfg.n_data = maze_data_train.mazes_onehot.shape[0]

    loaded = False
    if cfg.load:
        try:
            model, opt, logger = load(model, opt, cfg)
            loaded = True
        except FileNotFoundError as e:
            print("Failed to load, with error:\n", e)
            if cfg.evaluate:
                print("Skipping evaluation.")
                return
            else:
                print("Attempting to start experiment from scratch (not overwriting).")

        if cfg.evaluate:
            evaluate(model, maze_data_train, cfg.val_batch_size, "train", cfg, is_eval=True)
            evaluate(model, maze_data_test, cfg.val_batch_size, "test", cfg, is_eval=True)
            return

    if not loaded:
        if cfg.overwrite and os.path.exists(cfg.log_dir):
            shutil.rmtree(cfg.log_dir)
        try:
            os.mkdir(cfg.log_dir)
            logger = Logger()
        except FileExistsError as e:
            raise FileExistsError(f"Experiment log folder {cfg.log_dir} already exists. Use `--load` or `--overwrite` command line arguments "\
            "to load or overwrite it.")

    if cfg.wandb:
        hyperparam_cfg = {k: v for k, v in vars(cfg).items() if k not in set({'log_dir', 'exp_name'})}
        wandb.login()
        wandb.init(
            project='pathfinding-nca', 
            name=cfg.exp_name, 
            id=cfg.exp_name,
            config=hyperparam_cfg,
            resume="allow" if cfg.load else None,
            # resume="allow",
        )

    # Save a dictionary of the config to a json for future reference.
    # json.dump(cfg.__dict__, open(f'{cfg.log_dir}/config.json', 'w'))
    yaml.dump(OmegaConf.to_yaml(cfg), open(f'{cfg.log_dir}/config.yaml', 'w'))

    mazes_onehot, mazes_discrete, maze_ims, target_paths = \
        (maze_data_train.mazes_onehot, maze_data_train.mazes_discrete, maze_data_train.maze_ims, maze_data_train.target_paths)

# fig, ax = plt.subplots(figsize=(20, 5))
# plt.imshow(np.hstack(maze_ims[:cfg.render_minibatch_size]))

    assert cfg.path_chan == mazes_discrete.max() + 1  # Wait we don't even use this??

# solved_mazes = mazes_discrete.clone()
# solved_mazes[th.where(target_paths == 1)] = path_chan
# fig, ax = plt.subplots(figsize=(20, 5))
# plt.imshow(np.hstack(solved_maze_ims[:cfg.render_minibatch_size]))

# fig, ax = plt.subplots(figsize=(20, 5))
# plt.imshow(np.hstack(target_paths[:cfg.render_minibatch_size].cpu()))
# plt.tight_layout()

    assert cfg.n_in_chan == mazes_onehot.shape[1]

    # The set of initial mazes (padded with 0s, channel-wise, to act as the initial state of the CA).

    if cfg.render:
        with th.no_grad():
            render_trained(model, maze_data_train, cfg)
    else:
        train(model, opt, maze_data_train, maze_data_val, target_paths, logger, cfg)

    if cfg.wandb:
        wandb.finish()


if __name__ == '__main__':
    main_experiment()