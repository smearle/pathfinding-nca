import argparse
import os
from pdb import set_trace as TT

import torch


class ImmutableConfig():
    # The channel of the maze corresponding to empty/traversable space.
    empty_chan, wall_chan, src_chan, trg_chan = 0, 1, 2, 3

    n_in_chan = 4    # The number of channels in the one-hot encodings of the training mazes.

class Config():
    """Default configuration. Note that all of these static variables are treated as command line arguments via the 
    `ClArgsConfig` class below."""
    # The name of the experiment.
    exp_name = "0"

    # How many mazes on which to train at a time
    n_data = 256

    # The width (and height, mazes are square for now) of the maze.
    width, height = 16, 16

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
    n_hid_nodes = 256

    # How often to print results to the console.
    log_interval = 500

    # How often to save the model and optimizer to disk.
    save_interval = 2000

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
    shared_weights = True

    # Originally this was a bug. But the shared-weight NCA still converges? Also ~3x faster as backprop becomes much 
    # cheaper. Adding as cl arg to investigate further. I think this causes only the last layer to be updated?
    sparse_update = False

    # When to compute the loss and update the network. If None, only compute the loss at the end of an `n_layers`-length
    # episode.
    loss_interval = None

    # Whether to feed in the input maze at each pass through the network.
    skip_connections = True

    # Whether to zero out weights and gradients so as to ignore the corners of each 3x3 patch when using convolutions.
    cut_conv_corners = False

    test = False


class ClArgsConfig(Config, ImmutableConfig):

    def __init__(self, args=None):
        """Set default arguments and get command line arguments."""
        # Load default arguments.
        cfg = Config()
        immutable_cfg = ImmutableConfig()
        [setattr(self, k, getattr(cfg, k)) for k in dir(cfg) if not k.startswith('_')]

        # Get command-line arguments.
        if args is None:
            args = argparse.ArgumentParser()
        args.add_argument('--load', action='store_true')
        args.add_argument('--render', action='store_true')
        args.add_argument('--overwrite', action='store_true')

        # Include all parameters in the Config class as command-line arguments.
        for k, v in vars(self).items():
            if k.startswith('_'):
                continue
            if type(v) is bool:
                args.add_argument('--' + k, action=argparse.BooleanOptionalAction)
                continue
            if type(v) is None: 
                typ = int
            else:
                typ = type(v)
            args.add_argument(f'--{k}', type=typ, default=v)

        args = args.parse_args()

        # Command-line arguments will overwrite default config attributes.
        [setattr(self, k, v) for k, v in vars(args).items()]

        # Immutable config settings are not available as command-line arguments.
        [setattr(self, k, getattr(immutable_cfg, k)) for k in dir(immutable_cfg) if not k.startswith('_')]
 
    def set_exp_name(self):
        self._validate()
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
        self.log_dir = os.path.join("runs", self.exp_name)

    def _validate(self):
        self.n_hid_chan = 2 if self.model == "FixedBfsNCA" else self.n_hid_chan
        self.load = True if self.render else self.load
        # self.minibatch_size = 1 if self.model == "GCN" else self.minibatch_size
        self.val_batch_size = 1 if self.model == "GCN" else self.val_batch_size
        assert self.n_val_data % self.val_batch_size == 0, "Validation dataset size must be a multiple of val_batch_size."
        if self.sparse_update:
            assert self.shared_weights, "Sparse update only works with shared weights. (Otherwise early layers may not "\
                "be updated.)"
        if self.model == "GCN":
            self.cut_conv_corners = True
        if self.cut_conv_corners and not self.model == "GCN":
            assert self.model == "NCA", "Cutting corners only works with NCA."
        if self.loss_interval is None:
            self.loss_interval = self.n_layers
        
        # For backward compatibility, we assume 50k updates, where each update is following a 32-batch of episodes. So
        # we ensure that we have approximately the same number of episodes given different batch sizes here.
        if self.minibatch_size != 32:
            self.n_updates = int(self.n_updates * 32 / self.minibatch_size) 

        assert self.n_layers % self.loss_interval == 0, "loss_interval should divide n_layers."

class BatchConfig(ClArgsConfig):
    """A class for batch configurations. This is used for parallel SLURM jobs, or a sequence of local jobs."""
    def __init__(self):
        args = argparse.ArgumentParser()
        # These arguments apply only to the batch of experiments, and will not be considered by individual experiments 
        # themselves.
        args.add_argument('--slurm', action='store_true', help='Submit jobs to run in parallel on SLURM.')
        args.add_argument('--vis_cross_eval', action='store_true')
        args.add_argument('--gen_new_data', action='store_true', help="Generate new data on which to run the batch"
                            " of experiments.")
        super().__init__(args)


