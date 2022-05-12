import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
from pdb import set_trace as TT
from tokenize import String
from typing import Any, List, Optional
from xml.dom.pulldom import default_bufsize

import hydra
from hydra.core.config_store import ConfigStore
import torch


@dataclass
class Config():
    """Default configuration. For use with hydra."""
    # The channel of the maze corresponding to empty/traversable space.
    empty_chan, wall_chan, src_chan, trg_chan = 0, 1, 2, 3  # immutable

    # The number of channels in the one-hot encodings of the training mazes.
    n_in_chan = 4  # immutable

    # The name of the experiment.
    exp_name: str = "0"

    # How many mazes on which to train at a time
    n_data: int = 8192

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
    log_interval: int = 500

    # How often to save the model and optimizer to disk.
    save_interval: int = 2000

    eval_interval: int = 100

    # How many updates to perform during training.
    n_updates: int = 50000        

    # How many mazes on which to train at once
    minibatch_size: int= 32 

    # Size of minibatch for rendering images and animations.
    render_minibatch_size: int = 1

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate: float= 1e-4

    # Train on new random mazes. If None, train only on the initial set of generated mazes. 
    gen_new_data_interval: Optional[bool] = None

    # Which model to load/train.
    model: str = "NCA"

    # Whether the model is a cellular automaton-type model.
    shared_weights: bool = True

    # Originally this was a bug. But the shared-weight NCA still converges? Also ~3x faster as backprop becomes much 
    # cheaper. Adding as cl arg to investigate further. I think this causes only the last layer to be updated?
    sparse_update: bool = False

    # When to compute the loss and update the network. If None, only compute the loss at the end of an `n_layers`-length
    # episode.
    loss_interval: Optional[int] = None

    # Whether to feed in the input maze at each pass through the network.
    skip_connections: bool = True

    # Whether to zero out weights and gradients so as to ignore the corners of each 3x3 patch when using convolutions.
    cut_conv_corners: bool = False

    evaluate: bool = False
    load: bool = False
    render: bool = False
    overwrite: bool = False
    wandb: bool = True

    def set_exp_name(self):
        self.validate()
        # TODO: create distinct `full_exp_name` attribute where we'll write this thing, so we don't overwrite the user-
        # supplied `exp_name`.
        self.exp_name = ''.join([
            f"{self.model}",
            f"_shared-{('T' if self.shared_weights else 'F')}",
            f"_skip-{('T' if self.skip_connections else 'F')}",
            f"_{self.n_hid_chan}-hid",
            f"_lr-{'{:.0e}'.format(self.learning_rate)}",
            f"_{self.n_data}-data",
            f"_{self.n_layers}-layer",
            (f"_{self.loss_interval}-loss" if self.loss_interval != self.n_layers else ''),
            f"_cutCorners-{('T' if self.cut_conv_corners else 'F')}",
            f"_sparseUpdate-{('T' if self.sparse_update else 'F')}",
            f"_{self.exp_name}",

        ])
        self.log_dir = os.path.join(Path(__file__).parent.parent, "runs", self.exp_name)

    def validate(self):
        if self.model == "FixedBfsNCA":
            self.n_hid_chan = 7
            self.skip_connections = True
        self.load = True if self.render else self.load
        # self.minibatch_size = 1 if self.model == "GCN" else self.minibatch_size
        # self.val_batch_size = 1 if self.model == "GCN" else self.val_batch_size
        assert self.n_val_data % self.val_batch_size == 0, "Validation dataset size must be a multiple of val_batch_size."
        if self.sparse_update:
            assert self.shared_weights, "Sparse update only works with shared weights. (Otherwise early layers may not "\
                "be updated.)"
        if self.model == "GCN":
            self.cut_conv_corners = True
        elif self.cut_conv_corners:
            assert self.model == "NCA", "Cutting corners only works with NCA (optional) or GCN (forced)."
        if self.loss_interval is None:
            self.loss_interval = self.n_layers
        
        # For backward compatibility, we assume 50k updates, where each update is following a 32-batch of episodes. So
        # we ensure that we have approximately the same number of episodes given different batch sizes here.
        if self.minibatch_size != 32:
            self.n_updates = int(self.n_updates * 32 / self.minibatch_size) 

        assert self.n_layers % self.loss_interval == 0, "loss_interval should divide n_layers."
        if self.minibatch_size < 32:
            assert 32 % self.minibatch_size == 0, "minibatch_size should divide 32."
            self.n_updates = self.n_updates * 32 // self.minibatch_size

@dataclass
class HyperSweepConfig():
    """A config defining hyperparameter sweeps, whose cartesian product defines a set of `Config` instances."""
    model: List[Any] = field(default_factory=lambda: [
        "GCN",
    ])
    

@dataclass
class ModelSweep(HyperSweepConfig):
    model: List[Any] = field(default_factory=lambda: [
        "GCN"
    ])

    n_hid_chan: List[Any] = field(default_factory=lambda: [
        4,
        8,
        # 96,
        # 128,
        # 256,
    ])


@dataclass
class BatchConfig(Config):
    """A class for batch configurations. This is used for parallel SLURM jobs, or a sequence of local jobs."""
    sweep: HyperSweepConfig = ModelSweep()
    # batch_hyperparams: HyperSweepConfig = HyperSweepConfig()
    slurm: bool = False
    vis_cross_eval: bool = False
    gen_new_data: bool = False
    load_all: bool = False
    filter_by_config: bool = False
    selective_table: bool = False
    load_pickle: bool = False
    n_updates: int = 1000


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="batch_config", node=BatchConfig)
cs.store(group="sweep", name="default", node=HyperSweepConfig)
cs.store(group="sweep", name="models", node=ModelSweep)
