import torch


class PathfindingNN(torch.nn.Module):
    def __init__(self, n_in_chan=4, n_hid_chan=9, drop_diagonals=True):
        """A Neural Network for pathfinding.
        
        The underlying state of the maze is given through `reset()`, then concatenated with the input at all subsequent
        passes through the network until the next reset.
        """
        super().__init__()

        # The initial/actual state of the board (so that we can use walls to simulate path-flood).
        self.x0 = None

    def forward(self, x):
        # Concatenate the underlying state of the maze with the input.
        x = torch.cat([self.x0, x], dim=1)

        return x

    def reset(self, x0):
        self.x0 = x0
