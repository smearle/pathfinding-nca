from pdb import set_trace as TT

import torch
from torch import nn
import torch.nn.functional as F

from models.nn import PathfindingNN


class MLP(PathfindingNN):
  def __init__(self, cfg):
    """A Neural Cellular Automata model for pathfinding over grid-based mazes.
    
    Args:
      n_in_chan: Number of input channels in the onehot-encoded input.
      n_hid_chan: Number of channels in the hidden layers.
      drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
      """
    super().__init__()
    self.n_layers = cfg.n_layers

    # The size of the maze when flattened
    n_input = (cfg.width + 2) ** 2 * (cfg.n_in_chan)

    self.linears = nn.ModuleList([nn.Linear(n_input, cfg.n_hidden)])
    for _ in range(cfg.n_layers - 2):
        self.linears.append(nn.Linear(cfg.n_hidden, cfg.n_hidden))
    self.linears.append(nn.Linear(cfg.n_hidden, n_input))


  def forward(self, input, update_rate=0.5):
    # TODO: implement this for MLPs.
    # x = super().add_initial_maze(x)  

    print(input.shape)
    x = input.view(input.shape[0], -1).float()
    print(x.shape)

    for i in range(self.n_layers):
        print(x.shape)
        x = self.linears[i](x)
        x = F.relu(x)

    y = x.view(*input.shape)

    return y

  def reset(self, x0):
      pass
    # self.x0 = x0


