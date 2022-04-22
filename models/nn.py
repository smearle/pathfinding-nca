from abc import ABC

import torch
from torch import nn


class PathfindingNN(ABC, nn.Module):
    def __init__(self, n_in_chan=4, n_hid_chan=9, drop_diagonals=True):
        """A Neural Network for pathfinding.
        
        The underlying state of the maze is given through `reset()`, then concatenated with the input at all subsequent
        passes through the network until the next reset.
        """
        nn.Module.__init__(self)

        # The initial/actual state of the board (so that we can use walls to simulate path-flood).
        self.initial_maze = None

    def add_initial_maze(self, x):
        # Concatenate the underlying state of the maze with the input along the channel dimension.
        x = torch.cat([self.initial_maze, x], dim=1)

        return x

    def forward(self, x):
        raise NotImplementedError

    def reset(self, initial_maze):
        """Store the initia maze to concatenate with later activations."""
        self.initial_maze = initial_maze
