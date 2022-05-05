from pdb import set_trace as TT

import torch
from torch import nn
import torch.nn.functional as F
from config import ClArgsConfig

from models.nn import PathfindingNN


class MLP(PathfindingNN):
    def __init__(self, cfg: ClArgsConfig):
        """A Multilayer Perceptron for pathfinding over grid-based mazes.
        
        Args:
            n_in_chan: Number of input channels in the onehot-encoded input.
            n_hid_chan: Number of channels in the hidden layers.
            drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
            """
        super().__init__(cfg)
        self.n_layers = cfg.n_layers
        self.n_hid_chan = cfg.n_hid_chan

        # Reserve some channels for concatenating the input maze if using skip connectiions.
        n_out_chan = cfg.n_hid_chan if cfg.skip_connections else cfg.n_hid_chan + cfg.n_in_chan

        self.n_hid_nodes = cfg.n_hid_nodes
        self.n_input_nodes = (cfg.width + 2) * (cfg.height + 2) * (cfg.n_in_chan + cfg.n_hid_chan)
        self.n_out_nodes = (cfg.width + 2) * (cfg.height + 2) * n_out_chan

        # The size of the maze when flattened


        def _make_dense_sequence():
            return nn.Sequential(
                nn.Flatten(),
                # nn.Linear(self.n_input_nodes, self.n_out_nodes), nn.ReLU(),
                nn.Linear(self.n_input_nodes, self.n_hid_nodes), nn.ReLU(),
                nn.Linear(self.n_hid_nodes, self.n_out_nodes), nn.ReLU(),
                Reshape(shape=(n_out_chan, cfg.width+2, cfg.height+2)),
            )


        if cfg.shared_weights:
            dense_0 = _make_dense_sequence()
            modules = [dense_0 for _ in range(self.n_layers)]
        else:
            modules = [_make_dense_sequence() for _ in range(self.n_layers)]

        self.layers = nn.ModuleList(modules)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

