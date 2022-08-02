from importlib.resources import path
from pdb import set_trace as TT
from numpy import require

import torch as th
from torch import nn
from mazes import Tiles

from models.nn import PathfindingNN


th.set_printoptions(linewidth=200)


class FixedBfsNCA(PathfindingNN):
    adjs = [(0, 1), (1, 0), (1, 2), (2, 1)]
    # source and age for activation map; path channel and 4 directional path activation channels for path extraction
    n_hid_chan = 2 + 1 + len(adjs)

    def __init__(self, cfg, requires_grad=False):
        """A Neural Cellular Automata model for pathfinding over grid-based mazes.
        
        Args:
            n_in_chan: Number of input channels in the onehot-encoded input.
            n_hid_chan: Number of channels in the hidden layers.
            drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
            """
        super().__init__(cfg)
        self.src_chan = Tiles.SRC
        self.trg_chan = Tiles.TRG
        self.n_in_chan = cfg.n_in_chan
        # self.n_hid_chan = cfg.n_hid_chan
        assert self.n_in_chan == 4
        self.conv_0 = nn.Conv2d(self.n_in_chan + self.n_hid_chan, self.n_in_chan + self.n_hid_chan, 
            3, 1, padding=1, bias=True)
        # with th.no_grad():
        # input: (empty, wall, src, trg)
        # weight: (out_chan, in_chan, w, h)
        self.flood_chan = flood_chan = self.n_in_chan

        # This convolution handles the flood and path extraction.
        conv_0_weight = th.zeros_like(self.conv_0.weight)
        conv_0_bias = th.zeros_like(self.conv_0.bias)

        # the first n_in_chans channels will hold the actual map (via additive skip connections)

        # the next channel will contain the (binary) flood, with activation flowing from the source and flooded tiles...
        conv_0_weight[flood_chan, self.src_chan, 1, 1] = 1.
        for adj in self.adjs + [(1, 1)]:
            conv_0_weight[flood_chan, flood_chan, adj[0], adj[1]] = 1.

        # ...but stopping at walls.
        conv_0_weight[flood_chan, Tiles.WALL, 1, 1] = -6.

        # the next channel will contain the age of the flood
        self.age_chan = age_chan = flood_chan + 1
        conv_0_weight[age_chan, flood_chan, 1, 1] = 1.
        conv_0_weight[age_chan, age_chan, 1, 1] = 1.

        # in the final channel, we will extract the optimal path(s)
        self.path_chan = path_chan = age_chan + 1

        # The transposed convolution will be begin illuminating path channels along the shortest path once the flood
        # meets the target.
        conv_0_weight[path_chan, flood_chan, 1, 1] = 1.
        conv_0_weight[path_chan, self.trg_chan, 1, 1] = 1.
        conv_0_bias[path_chan] = -1.

        # It then spreads path channel to neighboring cells with greater age values.
        for i, adj in enumerate(self.adjs):
            conv_0_weight[path_chan + i + 1, path_chan, adj[0], adj[1]] = 1
            conv_0_weight[path_chan + i + 1, age_chan, adj[0], adj[1]] = 2
            conv_0_weight[path_chan + i + 1, age_chan, 1, 1] = -2

        self.conv_0.weight = nn.Parameter(conv_0_weight, requires_grad=requires_grad)
        self.conv_0.bias = nn.Parameter(conv_0_bias, requires_grad=requires_grad)
                
    def forward(self, x):
        # with th.no_grad():
        x = self.add_initial_maze(x)
        x = self.hid_forward(x)
        self.n_step += 1

        return x

    def hid_forward(self, x):
        self.n_batches = n_batches = x.shape[0]
        # agent_pos = (input.shape[2] // 2, input.shape[3] // 2)
        trg_pos = th.where(x[:, self.trg_chan, ...] == 1)
        if trg_pos[0].shape[0] == 0:
            trg_pos = th.zeros(n_batches, 3, dtype=th.int64)
            trg_pos = tuple([trg_pos[:, i] for i in range(3)])
        # trg_pos = (trg_pos[1].item(), trg_pos[2].item())
        # batch_dones = self.get_dones(x, trg_pos)
        # if not batch_dones.all().item():
        x = self.flood(x)
        # batch_cont = batch_dones == False
        # x[batch_cont] = x1[batch_cont]
        # self.batch_dones = batch_dones = self.get_dones(x, trg_pos)
        
        # Exclude the initial maze from the input (included for convenience)
        return x[:, self.n_in_chan:]
            
    def flood(self, x):
        # x0 = x.clone()  # for debugging
        x = self.conv_0(x)

        # x[:, self.flood_chan] = th.clamp(x[:, self.flood_chan], 0., 1.)
        x[:, self.flood_chan] = clamp(x[:, self.flood_chan], 0., 1.)

        # Are the directional path activation channels equal to -1?
        # path_activs = x[:, self.path_chan + 1: self.path_chan + 1 + len(self.adjs)] == -1.0
        path_activs = sawtooth_relu(x[:, self.path_chan + 1: self.path_chan + 1 + len(self.adjs)], -1)
        # path_activs = th.max(path_activs, dim=1)[0]
        path_activs = clamp(th.sum(path_activs, dim=1), 0, 1)
        # path_activs = clamp_relu(path_activs, 0, 1)[0]
        # If any one of them is, then make the path channel at that cell equal to 1.
        x[:, self.path_chan] += path_activs.float()
        x[:, self.path_chan] = clamp(x[:, self.path_chan], 0., 1.)
        return x

    def get_dones(self, x, trg_pos):
        batch_dones = x[trg_pos[0], self.age_chan, trg_pos[1], trg_pos[2]] > 0.1
        return batch_dones

    def reset(self, initial_maze, is_torchinfo_dummy=False, **kwargs):
        super().reset(initial_maze)


def clamp_relu(x, min_val, max_val):
    """Essentially `th.clamp(x, min_val, max_val)`, but differentiable."""
    return -th.relu(-(th.relu(x - min_val) + min_val) + max_val) + max_val


def sawtooth_relu(x, a):
    """Returns 1 when x = a. Assume slope is 1. TODO: make this an argument, `m`."""
    # The upward slope. This is 0 when x <= a - 1
    return th.relu(x - a + 1) - 2 * th.relu(x - a) + th.relu(x - a - 1)


clamp = clamp_relu
# clamp = th.clamp