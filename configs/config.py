import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
from pdb import set_trace as TT
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
import torch as th

from configs.sweeps.all import HyperSweepConfig
from configs.sweeps.all_scratch import ScratchSweep
from configs.sweeps.cut_corners import CutCornerSweep
from configs.sweeps.gnn import GNNSweep
from configs.sweeps.handcoded import HandcodedSweep
from configs.sweeps.kernel import KernelSweep
from configs.sweeps.maxpool import MaxPoolSweep
from configs.sweeps.evo_data_scratch import EvoDataScratchSweep
from configs.sweeps.loss_interval import LossIntervalSweep
from configs.sweeps.models import ModelSweep
from configs.sweeps.n_hid_chan import HidChanSweep
from configs.sweeps.evo_data import EvoDataSweep
from configs.sweeps.weight_sharing import WeightSharingSweep
from configs.sweeps.gnn import GNNSweep


@dataclass
class Config():
    """Default configuration. For use with hydra."""
    # The number of channels in the one-hot encodings of the training mazes.
    n_in_chan = 4  # cannot be controlled by user

    # The name of the experiment.
    exp_name: str = "0"

    # How many mazes on which to train at a time
    n_data: int = 10000

    # The width (and height, mazes are square for now) of the maze.
    width: int = 16 
    height: int = 16

    # Size of validation dataset
    n_val_data: int = 8192
    val_batch_size: int = 64

    n_test_data: int = 8192

    # Number of steps after which we calculate loss and update the network.
    n_layers: int = 64    

    # The number in/out channels, including the input channels (n_in_chan)
    # n_aux_chan = 48    

    # The number of channels in the hidden layers. (For NCA and GCA models.)
    n_hid_chan: int = 96    

    # The number of hidden nodes in the hidden layers. (For MLP models.)
    n_hid_nodes: int = 256

    # How often to print results to the console.
    log_interval: int = 1000

    # How often to save the model and optimizer to disk.
    save_interval: int = 2000

    eval_interval: int = 100

    # How many updates to perform during training.
    n_updates: int = 50000        

    # How many mazes on which to train at once
    minibatch_size: int= 32 

    # Size of minibatch for rendering images and animations.
    render_minibatch_size: int = 1

    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cuda'

    learning_rate: float= 1e-4

    # Which model to load/train.
    model: str = "NCA"

    # Whether the model is a cellular automaton-type model.
    shared_weights: bool = True

    # Originally this was a bug. But the shared-weight NCA still converges? Also ~3x faster as backprop becomes much 
    # cheaper. Adding as cl arg to investigate further. I think this causes only the last layer to be updated?
    sparse_update: bool = False

    # When to compute the loss and update the network. If None, only compute the loss at the end of an `n_layers`-length
    # episode.
    loss_interval: Optional[Any] = None

    # Whether to feed in the input maze at each pass through the network.
    skip_connections: bool = True

    # Whether to zero out weights and gradients so as to ignore the corners of each 3x3 patch when using convolutions.
    cut_conv_corners: bool = False

    # When using NCA, encusre the weights for each adjacent cell are the same, making it essentially equivalent to a GCN.
    symmetric_conv: bool = False

    # Whether to dedicate some channels in the NCA to a per-layer max-pooling operation (one spatial, one channel-wise,
    # over whole map and all channels, respectively).
    max_pool: bool = False

    # What size kernel to use in convolutioanl layers (when using an NCA model).
    kernel_size: int = 3

    task: str = "pathfinding"
    evaluate: bool = False
    load: bool = False
    render: bool = False
    overwrite: bool = False
    wandb: bool = False

    # Update our own image of loss curves and model outputs in the training directory (in addition to writing them to 
    # tensorboard and/or wandb).
    manual_log: bool = False

    # A regime for generating environments in parallel with training with the aim of increasing the model's generality.
    env_generation: Any = None

    # Load a config from a file at this path. Will override all other config options.
    load_config_path: Any = None

    ### For GNNs only -- grid to graph conversion ###
    # TODO: Make this a separate sub-config, active only when we are using GNNs?
    # If False, feed the GNN the entire grid. If True, feed it only traversable nodes and edges.
    traversable_edges_only: bool = False
    # If True, include edge features corresponding to the edge's relative position to the central node.
    positional_edge_features: bool = False
    ###


@dataclass
class BatchConfig(Config):
    """A class for batch configurations. This is used for parallel SLURM jobs, or a sequence of local jobs."""
    sweep: HyperSweepConfig = HyperSweepConfig()
    # batch_hyperparams: HyperSweepConfig = HyperSweepConfig()
    slurm: bool = False
    vis_cross_eval: bool = False
    gen_new_data: bool = False
    load_all: bool = False
    filter_by_config: bool = False
    # selective_table: bool = False
    load_pickle: bool = False
    n_updates: int = 50000
    overwrite_evals: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="batch_config", node=BatchConfig)
cs.store(group="sweep", name="default", node=HyperSweepConfig)
cs.store(group="sweep", name="models", node=ModelSweep)
cs.store(group="sweep", name="gnn", node=GNNSweep)
cs.store(group="sweep", name="loss_interval", node=LossIntervalSweep)
cs.store(group="sweep", name="n_hid_chan", node=HidChanSweep)
cs.store(group="sweep", name="evo_data", node=EvoDataSweep)
cs.store(group="sweep", name="cut_corners", node=CutCornerSweep)
cs.store(group="sweep", name="max_pool", node=MaxPoolSweep)
cs.store(group="sweep", name="kernel", node=KernelSweep)
cs.store(group="sweep", name="shared_weights", node=WeightSharingSweep)
cs.store(group="sweep", name="handcoded", node=HandcodedSweep)

cs.store(group="sweep", name="scratch", node=ScratchSweep)
cs.store(group="sweep", name="evo_data_scratch", node=EvoDataScratchSweep)

