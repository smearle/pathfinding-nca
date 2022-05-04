from abc import ABC, abstractmethod
from pdb import set_trace as TT

import torch
from torch import nn


class PathfindingNN(ABC, nn.Module):
    def __init__(self, n_in_chan=4, n_hid_chan=9, skip_connections=True):
        """A Neural Network for pathfinding.
        
        The underlying state of the maze is given through `reset()`, then concatenated with the input at all subsequent
        passes through the network until the next reset.
        """
        nn.Module.__init__(self)
        self.n_step = 0
        self.skip_connections = skip_connections

        # The initial/actual state of the board (so that we can use walls to simulate path-flood).
        self.initial_maze = None

    def add_initial_maze(self, x):
        # Concatenate the underlying state of the maze with the input along the channel dimension.
        x = torch.cat([self.initial_maze, x], dim=1)

        return x

    def forward(self, x):
        # This forward pass will iterate through a single layer (or all layers if passing dummy input via `torchinfo`).
        for _ in range(1 if not self.is_torchinfo_dummy else len(self.layers)):
            if self.skip_connections or self.n_step == 0:
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
