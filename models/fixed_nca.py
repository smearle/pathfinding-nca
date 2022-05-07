from pdb import set_trace as TT

import torch as th
from torch import nn

from models.nn import PathfindingNN


adjs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
adjs_to_acts = {adj: i for i, adj in enumerate(adjs)}


class FixedBfsNCA(PathfindingNN):
    n_hid_chan = 3

    def __init__(self, cfg):
        """A Neural Cellular Automata model for pathfinding over grid-based mazes.
        
        Args:
            n_in_chan: Number of input channels in the onehot-encoded input.
            n_hid_chan: Number of channels in the hidden layers.
            drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
            """
        super().__init__(cfg)
        self.src_chan = cfg.src_chan
        self.trg_chan = cfg.trg_chan
        self.n_in_chan = cfg.n_in_chan
        # self.n_hid_chan = cfg.n_hid_chan
        assert self.n_in_chan == 4
        self.conv_0 = nn.Conv2d(self.n_in_chan + self.n_hid_chan, self.n_in_chan + self.n_hid_chan, 
            3, 1, padding=1, bias=False)
        self.conv_1 = nn.ConvTranspose2d(self.n_in_chan + self.n_hid_chan, self.n_in_chan + self.n_hid_chan,
            3, 1, padding=1, bias=False)
        with th.no_grad():
            # input: (empty, wall, src, trg)
            # weight: (out_chan, in_chan, w, h)

            # ... and copies the source to a flood tile
            self.flood_chan = flood_chan = self.n_in_chan
            # self.conv_0.weight[flood_chan, self.src_chan, 0, 0] = 1

            # this convolution handles the flood
            self.conv_0.weight = nn.Parameter(th.zeros_like(self.conv_0.weight), requires_grad=False)

            # the first n_in_chans channels will hold the actual map (via additive skip connections)

            # the next channel will contain the (binary) flood, with activation flowing from the source and flooded tiles...
            for adj in adjs:
                self.conv_0.weight[flood_chan, self.src_chan, adj[0], adj[1]] = 1.
                self.conv_0.weight[flood_chan, flood_chan, adj[0], adj[1]] = 1.

            # ...but stopping at walls.
            self.conv_0.weight[flood_chan, cfg.wall_chan, 1, 1] = -6.

            # the next channel will contain the age of the flood
            self.age_chan = age_chan = flood_chan + 1
            self.conv_0.weight[age_chan, flood_chan, 1, 1] = 1.
            self.conv_0.weight[age_chan, age_chan, 1, 1] = 1.

            # in the final channel, we will extract the optimal path(s)
            self.path_chan = path_chan = age_chan + 1
            # self.conv_1.weight[]
                
    def forward(self, x0):
        x = self.add_initial_maze(x0)
        with th.no_grad():
            x = self.hid_forward(x)

        return x

    def hid_forward(self, x):
        self.n_batches = n_batches = x.shape[0]
        # agent_pos = (input.shape[2] // 2, input.shape[3] // 2)
        trg_pos = th.where(x[:, self.trg_chan, ...] == 1)
        if trg_pos[0].shape[0] == 0:
            trg_pos = th.zeros(x.shape[0], 3, dtype=th.int64)
            trg_pos = tuple([trg_pos[:, i] for i in range(3)])
        # trg_pos = (trg_pos[1].item(), trg_pos[2].item())
        batch_dones = self.get_dones(x, trg_pos)
        if not batch_dones.all():
            x1 = self.flood(x)
            batch_cont = batch_dones == False
            x[batch_cont] = x1[batch_cont]
            self.batch_dones = batch_dones = self.get_dones(x, trg_pos)
        
        # Exclude the initial maze from the input (included for convenience)
        return x[:, self.n_in_chan:]
            
    def flood(self, x):
        x = self.conv_0(x)
        x[:, self.flood_chan] = th.clamp(x[:, self.flood_chan], 0., 1.)
        # x[:, :self.n_in_chan] += input

        y = self.conv_1(x)
        TT()
        return x

    def get_solution_length(self, input):
        x = self.hid_forward(input)
        if not self.batch_dones.all():
            return 0
        return self.i

    def get_dones(self, x, agent_pos):
        batch_dones = x[agent_pos[0], self.age_chan, agent_pos[1], agent_pos[2]] > 0.1
        return batch_dones

    def reset(self, initial_maze, is_torchinfo_dummy=False, **kwargs):
        super().reset(initial_maze)

