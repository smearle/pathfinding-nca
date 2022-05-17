from pdb import set_trace as TT

import numpy as np
import torch as th
from torch import nn
from configs.config import Config

from models.nn import PathfindingNN


th.set_printoptions(threshold=1000, linewidth=1000)


class FixedDfsNCA(PathfindingNN):
    # Top, left, bottom, right neighbors.
    rel_adjs = np.array([[-1, 0], [0,-1], [1, 0], [0, 1]])

    adjs = rel_adjs + np.array([[2, 2]])
    # overall and directional floods, queued flood channels
    n_hid_chan = 12

    def __init__(self, cfg: Config):
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
            5, 1, padding=2, bias=True)
        with th.no_grad():
            # input: (empty, wall, src, trg)
            # weight: (out_chan, in_chan, w, h)
            self.flood_chan = flood_chan = self.n_in_chan
            self.pebble_chan = self.flood_chan + 5
            self.stack_chan = self.pebble_chan + 1

            # This convolution handles the flood and path extraction.
            self.conv_0.weight = nn.Parameter(th.zeros_like(self.conv_0.weight), requires_grad=False)
            self.conv_0.bias = nn.Parameter(th.zeros_like(self.conv_0.bias), requires_grad=False)

            # The first n_in_chans channels will hold the actual map (via additive skip connections)

            # The next channel will contain the (binary) flood, with activation flowing from the source and flooded tiles.
            self.conv_0.weight[flood_chan, self.src_chan, 2, 2] = 1.

            # Maintain separate channels to consider the floods coming in from each possible direction.
            for ai, adj in enumerate(self.adjs):

                # Flood should flow from this direction, unless it is flowing to one if its own higher-priority neighbors.
                self.conv_0.weight[flood_chan + 1 + ai, flood_chan, adj[0], adj[1]] = 1.

                # The relevant neighbor's neighbors. In order such that if the order of `adjs` is, e.g., top, right,
                # bottom, left, then the neighbors of the neighbors will be in the order bottom, left, top, right.
                b_adjs = adj - self.rel_adjs
                for bi, b_adj in enumerate(b_adjs[:ai]):

                    if np.all(b_adj == [2,2]):
                        continue
                    self.conv_0.weight[flood_chan + 1 + ai, cfg.empty_chan, b_adj[0], b_adj[1]] = -1.
                    self.conv_0.weight[flood_chan + 1 + ai, self.src_chan, b_adj[0], b_adj[1]] = -1.
                    self.conv_0.weight[flood_chan + 1 + ai, flood_chan, b_adj[0], b_adj[1]] = 1.
            
            # Flood is blocked by walls. (Also blocking directional floods for interpretability.)
            self.conv_0.weight[flood_chan: flood_chan + 5, cfg.wall_chan, 2, 2] = -2.

            self.conv_0.weight[flood_chan, self.stack_chan, 2, 2] = -1.

            # Flood is maintained once present at a given tile.
            self.conv_0.weight[flood_chan, flood_chan, 2, 2] = 1.

            # Stack chan becomes active when the pebble passes by us, but we are not flooded.
            self.conv_0.weight[self.stack_chan, self.flood_chan, 2, 2] = -1.
            for adj in self.adjs:
                self.conv_0.weight[self.stack_chan, self.pebble_chan, adj[0], adj[1]] = 1.

            # Stack chan stays activated.
            self.conv_0.weight[self.stack_chan, self.stack_chan, 2, 2] = 1.

            # Do not add wall tiles to stack
            self.conv_0.weight[self.stack_chan, cfg.wall_chan, 2, 2] = -1.

            # I need to be in the stack to be activated when the flood is stuck.

    def forward(self, x):
        with th.no_grad():
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
        x0 = x.clone()  # for debugging
        x = self.conv_0(x)
        x[:, self.flood_chan + 1: self.flood_chan + 5] = th.clamp(x[:, self.flood_chan + 1: self.flood_chan + 5], 0., 1.)
        adj_flood = th.max(x[:, self.flood_chan + 1: self.flood_chan + 5], dim=1)[0]
        x[:, self.flood_chan] += adj_flood
        x[:, self.flood_chan] = th.clamp(x[:, self.flood_chan], 0., 1.)

        # Pebble represents the edge of the flood (which only occupies one new tile per time slot).
        pbl = x[:, self.flood_chan] - x0[:, self.flood_chan]
        x[:, self.pebble_chan] = pbl

        x[:, self.stack_chan] = th.clamp(x[:, self.stack_chan], 0., 1.)

        return x

    def get_dones(self, x, trg_pos):
        batch_dones = x[trg_pos[0], self.age_chan, trg_pos[1], trg_pos[2]] > 0.1
        return batch_dones

    def reset(self, initial_maze, is_torchinfo_dummy=False, **kwargs):
        super().reset(initial_maze)

