from pdb import set_trace as TT

import torch


class NCA(torch.nn.Module):
  def __init__(self, n_in_chan=4, n_hidden_chan=9, drop_diagonals=True):
    """A Neural Cellular Automata model for pathfinding over grid-based mazes.
    
    Args:
      n_in_chan: Number of input channels in the onehot-encoded input.
      n_hidden_chan: Number of channels in the hidden layers.
      drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
      """
    super().__init__()
    # Number of hidden channels, also number of writable channels the the output. (None-maze channels at the input.)
    self.n_hidden_chan = n_hidden_chan  

    conv2d = torch.nn.Conv2d

    # This layer applies a dense layer to each 3x3 block of the input.
    self.l1 = conv2d(n_hidden_chan + n_in_chan, n_hidden_chan, kernel_size=3, padding=1)

    # Since the NCA receives the onehot maze as input at each step, we do not write to the channels reserved for this.
    self.l2 = conv2d(n_hidden_chan, n_hidden_chan, kernel_size=1) 

    # self.w2.weight.data.zero_()

    # The initial/actual state of the board (so that we can use walls to simulate path-flood).
    self.x0 = None

  def forward(self, x, update_rate=0.5):
    y = torch.cat([self.x0, x], dim=1)
    y = self.l1(y)
    y = torch.relu(y)
    y = self.l2(y)
    y = (torch.sigmoid(y) - 0.5) * 2
    b, c, h, w = y.shape

    return y

# def seed(self, n, sz=16):
#   return torch.zeros(n, self.chn, sz, sz)

  def reset(self, x0):
    self.x0 = x0

# def to_rgb(x):
  # return x[...,:3,:,:]+0.5
