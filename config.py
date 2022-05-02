import argparse
import os
from pdb import set_trace as TT

import torch


class Config():
    """Default configuration. Note that all of these static variables are treated as command line arguments via the 
    `ClArgsConfig` class below."""
    # The name of the experiment.
    exp_name = "0"

    # How many mazes on which to train at a time
    n_data = 256

    # The width (and height, mazes are square for now) of the maze.
    width = 16

    # Size of validation dataset
    n_val_data = 64
    val_batch_size = 64

    render_minibatch_size = 8    # how many mazes to render at once

    # Number of steps after which we calculate loss and update the network.
    n_layers = 64    

    # The number in/out channels, including the input channels (n_in_chan)
    # n_aux_chan = 48    

    # The number of channels in the hidden layers. (For NCA and GCA models.)
    n_hid_chan = 96    

    # The number of hidden nodes in the hidden layers. (For MLP models.)
    n_hidden = 256

    # How often to print results to the console.
    log_interval = 512

    # How often to save the model and optimizer to disk.
    save_interval = log_interval * 5    

    eval_interval = 100

    # How many updates to perform during training.
    n_updates = 50000        

    # How many mazes on which to train at once
    minibatch_size = 32 

    # Size of minibatch for rendering images and animations.
    render_minibatch_size = 8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = 1e-4

    # Train on new random mazes. If None, train only on the initial set of generated mazes. 
    gen_new_data_interval = None

    # Which model to load/train.
    model = "NCA"

    # Whether the model is a cellular automaton-type model.
    ca_model = True


class ClArgsConfig(Config):

    def __init__(self):
        """Set default arguments and get command line arguments."""
        # Load default arguments.
        cfg = Config()
        [setattr(self, k, getattr(cfg, k)) for k in dir(cfg) if not k.startswith('_')]

        # Get command-line arguments.
        args = argparse.ArgumentParser()
        args.add_argument('--load', action='store_true')
        args.add_argument('--render', action='store_true')
        args.add_argument('--overwrite', action='store_true')

        # Include all parameters in the Config class as command-line arguments.
        [args.add_argument('--' + k, type=type(v), default=v) for k, v in self.__dict__.items()]
        args = args.parse_args()

        # Command-line arguments will overwrite default config attributes.
        [setattr(self, k, v) for k, v in vars(args).items()]

        self.load = True if self.render else self.load
        self.minibatch_size = 1 if self.model == "GCN" else self.minibatch_size
        self.val_batch_size = 1 if self.model == "GCN" else self.val_batch_size
        assert self.n_val_data % self.val_batch_size == 0, "Validation dataset size must be a multiple of val_batch_size."
        
        # For now, only the MLP can be not cellular-automaton-like.
        # TODO: Allow this to be False for non-MLP models. Allow it to be true for MLP
        self.ca_model = self.model != "MLP"
        
        self.log_dir = get_exp_name(self)


def get_exp_name(cfg):
    exp_name = os.path.join("log", f"{cfg.model}_{cfg.n_hid_chan}-hid_{cfg.n_data}-data_{cfg.n_layers}-layer_{cfg.exp_name}")

    return exp_name

