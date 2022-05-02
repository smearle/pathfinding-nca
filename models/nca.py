from pdb import set_trace as TT

import torch

from models.nn import PathfindingNN


class NCA(PathfindingNN):
    def __init__(self, cfg):
        """A Neural Cellular Automata model for pathfinding over grid-based mazes.
        
        Args:
            n_in_chan: Number of input channels in the onehot-encoded input.
            n_hid_chan: Number of channels in the hidden layers.
            drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
            """
        super().__init__()
        n_in_chan, n_hid_chan = cfg.n_in_chan, cfg.n_hid_chan
        # Number of hidden channels, also number of writable channels the the output. (None-maze channels at the input.)
        self.n_hid_chan = n_hid_chan    

        conv2d = torch.nn.Conv2d

        # This layer applies a dense layer to each 3x3 block of the input.
        self.l1 = conv2d(n_hid_chan + n_in_chan, n_hid_chan, kernel_size=3, padding=1)

        # Since the NCA receives the onehot maze as input at each step, we do not write to the channels reserved for this.
        self.l2 = conv2d(n_hid_chan, n_hid_chan, kernel_size=1) 

        # self.w2.weight.data.zero_()

    def forward(self, x, update_rate=0.5):
        x = super().add_initial_maze(x)
        y = self.l1(x)
        y = torch.relu(y)
        y = self.l2(y)
        # y = torch.relu(y)
        # y = (torch.sigmoid(y) - 0.5) * 2
        # b, c, h, w = y.shape

        return y

# def seed(self, n, sz=16):
#     return torch.zeros(n, self.chn, sz, sz)

# def to_rgb(x):
    # return x[...,:3,:,:]+0.5
