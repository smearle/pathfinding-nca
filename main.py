#@title Imports and Notebook Utilities
#%tensorflow_version 2.x

import copy
from functools import partial
import json
import math
import os
from pdb import set_trace as TT
import pickle
import shutil
import sys
from matplotlib import animation

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch as th
import torchinfo
import wandb
import yaml

from configs import helper as config_helper
from configs.config import BatchConfig, Config
from evaluate import evaluate
from mazes import load_dataset
from models import BfsNCA, FixedBfsNCA, FixedDfsNCA, GAT, GCN, HillClimbingNCA, MLP, NCA
from models.fuzzy_pathfinding_nca import FuzzyPathfindingNCA
from models.nn import PathfindingNN
from render import render_trained
from train import train
from utils import Logger, VideoWriter, get_discrete_loss, log_stats, to_path, load


np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


def main_experiment_load_cfg(cfg_path: str = None):
    cfg = Config()
    load_cfg = OmegaConf.load(cfg_path)
    [setattr(cfg, k, v) for k, v in load_cfg.items() if not k.startswith('_')]
    config_helper.set_exp_name(cfg)
    main_experiment(cfg=cfg)


# @hydra.main(config_path=None, config_name="config")
def main_experiment(cfg: Config = None, cfg_path: str = None):
    # TODO: Re-enable running this script/function directly? (i.e. main hydra.main.) Currently always going through 
    #   `run_batch.py`.
    if cfg is None:
        cfg = Config()
    if not hasattr(cfg, 'full_exp_name'):
        cfg.set_exp_name()
    # Validate and set full experiment name if this has not been done already (i.e. if running this script directly or 
    # launching via SLURM).
    # if not hasattr(load_cfg, 'full_exp_name'):
        # cfg = Config()
        # [setattr(cfg, k, v) for k, v in vars(load_cfg).items() if not k.startswith('_')]
        # cfg.set_exp_name()
    os.system('nvidia-smi -L')
    if th.cuda.is_available():
            print('Using GPU/CUDA.')
            th.set_default_tensor_type('torch.cuda.FloatTensor')
            cfg.device = "cuda"
    else:
            print('Not using GPU/CUDA, using CPU.')
        
    model_cls = globals()[cfg.model]

    if cfg.task == 'pathfinding_solnfree':
        cfg.loss_fn = partial(cfg.loss_fn, pathfinding_model=HillClimbingNCA(cfg=cfg), cfg=cfg)
    elif cfg.task == 'maze_gen':
        cfg.loss_fn = partial(cfg.loss_fn, pathfinding_model=FuzzyPathfindingNCA(cfg=cfg), cfg=cfg)

    print(f"Running experiment with config:\n {OmegaConf.to_yaml(cfg)}")
    
    # Setup training.
    model: PathfindingNN = model_cls(cfg)
    # Set a dummy initial maze state to probe model. Model will default to edges for full grid.
    x0 = th.zeros(cfg.minibatch_size, cfg.n_in_chan, cfg.width + 2, cfg.width + 2)
    model.reset(x0, is_torchinfo_dummy=True)

    if not (cfg.render or cfg.evaluate):
        dummy_input = model.seed(batch_size=cfg.minibatch_size)
        torchinfo.summary(model, input_size=dummy_input.shape)

    # param_n = sum(p.numel() for p in model.parameters())
    # print('model param count:', param_n)
    opt = th.optim.Adam(model.parameters(), cfg.learning_rate)
    print("Loading data...")
    try:
        maze_data_train, maze_data_val, maze_data_test = load_dataset(cfg)
    except FileNotFoundError as e:
        print("No maze data files found. Run `python mazes.py` to generate the dataset.")
        raise
    print("Done loading data.")
    # maze_data_val.get_subset(cfg.n_val_data)
  # cfg.n_data = maze_data_train.mazes_onehot.shape[0]

    cfg_32 = copy.copy(cfg)
    cfg_32.width, cfg_32.height = 32, 32
    if cfg.shared_weights:
        cfg_32.n_layers = cfg.n_layers * 2
    maze_data_test_32 = load_dataset(cfg_32, test_only=True)
    print("Done loading 32x32 mazes.")

    loaded = False
    if cfg.load:
        try:
            model, opt, logger = load(model, opt, cfg)
            if cfg.env_generation:
                print("Loading environment generation data.")
                maze_data_train = pickle.load(open(f"{cfg.log_dir}/evo_mazes.pkl", "rb"))
            loaded = True
        except FileNotFoundError as e:
            print("Failed to load, with error:\n", e)
            if cfg.evaluate:
                print("Skipping evaluation.")
                return
            else:
                print("Attempting to start experiment from scratch (not overwriting).")

    if cfg.evaluate:
        log_stats(model, logger, cfg)
        evaluate(model, maze_data_train, cfg.val_batch_size, "train", cfg, is_eval=True)
        evaluate(model, maze_data_test, cfg.val_batch_size, "test", cfg, is_eval=True)
        if cfg.model != "MLP":
            evaluate(model, maze_data_test_32, cfg.val_batch_size, "test_32", cfg_32, is_eval=True)
        return

    if not loaded:
        if cfg.overwrite and os.path.exists(cfg.log_dir):
            shutil.rmtree(cfg.log_dir)
        try:
            os.mkdir(cfg.log_dir)
            logger = Logger()
        except FileExistsError as e:
            raise FileExistsError(f"Experiment log folder {cfg.log_dir} already exists. Use `load=True` or "
            "`overwrite=True` command line arguments to load or overwrite it.")

    if cfg.wandb:
        hyperparam_cfg = {k: v for k, v in vars(cfg).items() if k not in set({'log_dir', 'exp_name'})}
        wandb.login()
        wandb.init(
            project='pathfinding-nca', 
            name=cfg.full_exp_name, 
            id=cfg.full_exp_name,
            config=hyperparam_cfg,
            # resume="allow" if cfg.load else None,
        )

    # Save a dictionary of the config to a json for future reference.
    # json.dump(cfg.__dict__, open(f'{cfg.log_dir}/config.json', 'w'))
    yaml.dump(OmegaConf.to_yaml(cfg), open(f'{cfg.log_dir}/config.yaml', 'w'))

    mazes_onehot, mazes_discrete, maze_ims, target_paths = \
        (maze_data_train.mazes_onehot, maze_data_train.mazes_discrete, maze_data_train.maze_ims, maze_data_train.target_paths)

    # assert cfg.path_chan == mazes_discrete.max() + 1  # Wait we don't even use this??
    assert cfg.n_in_chan == mazes_onehot.shape[1]

    if cfg.render:
        with th.no_grad():
            render_trained(model, maze_data_train, cfg)
            render_trained(model, maze_data_test_32, cfg_32, name="_32")

    else:
        print("Beginning to train.")
        train(model, opt, maze_data_train, maze_data_val, maze_data_test_32, target_paths, logger, cfg, cfg_32)

    if cfg.wandb:
        wandb.finish()


if __name__ == '__main__':
    main_experiment()