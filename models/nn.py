from abc import ABC, abstractmethod
from pdb import set_trace as TT

import torch as th
from torch import nn

from config import ClArgsConfig


class PathfindingNN(ABC, nn.Module):
    def __init__(self, cfg: ClArgsConfig):
        """A Neural Network for pathfinding.
        
        The underlying state of the maze is given through `reset()`, then concatenated with the input at all subsequent
        passes through the network until the next reset.
        """
        nn.Module.__init__(self)
        self.n_step = 0
        self.cfg = cfg

        # The initial/actual state of the board (so that we can use walls to simulate path-flood).
        self.initial_maze = None

        # Reserve some channels for concatenating the input maze if using skip connectiions.
        self.n_out_chan = cfg.n_hid_chan + (cfg.n_in_chan if cfg.skip_connections else 0)


    def add_initial_maze(self, x):
        if self.cfg.skip_connections:
            # Concatenate the underlying state of the maze with the input along the channel dimension.
            x = th.cat([self.initial_maze, x], dim=1)
        else:
            # Overwrite the additional hidden channels with the underlying state of the maze.
            x[:, self.cfg.n_hid_chan: self.cfg.n_hid_chan + self.initial_maze.shape[1]] = self.initial_maze

        return x

    def forward(self, x):
        # This forward pass will iterate through a single layer (or all layers if passing dummy input via `torchinfo`).
        for _ in range(1 if not self.is_torchinfo_dummy else self.cfg.n_layers):
            if self.cfg.skip_connections or self.n_step == 0:
                x = self.add_initial_maze(x)
            x = self.forward_layer(x, self.n_step)
            self.n_step += 1

        return x

    def forward_layer(self, x, i):
        # assert hasattr(self, 'layers'), "Subclass of PathfindingNN must have `layers` attribute."
        x = self.layers[i](x)

        return x

    def reset(self, initial_maze, is_torchinfo_dummy=False):
        """Store the initia maze to concatenate with later activations."""
        self.is_torchinfo_dummy = is_torchinfo_dummy
        self.initial_maze = initial_maze
        self.n_step = 0

    def seed(self, batch_size):
        # NOTE: I think the effect of `sparse_update` here is to only update the last layer. Will break non-shared-weight
        #   networks. Should implement this more explicitly in the future.

        if self.cfg.skip_connections:
            n_chan = self.cfg.n_hid_chan
        else:
            n_chan = self.cfg.n_hid_chan + self.cfg.n_in_chan

        x = th.zeros(batch_size, n_chan, self.cfg.height + 2, self.cfg.width + 2, requires_grad=False)
        # x = th.zeros(batch_size, n_chan, cfg.height + 2, cfg.width + 2, requires_grad=not cfg.sparse_update)

        return x
