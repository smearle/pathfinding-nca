
from collections import OrderedDict
from pdb import set_trace as TT

import torch as th
from torch import nn
from configs.config import Config

from models.fixed_nca import FixedBfsNCA
from models.nca import NCA
from models.nn import PathfindingNN


th.autograd.set_detect_anomaly(True)


class BfsNCA(NCA):
    def __init__(self, cfg: Config):
        """A hybrid model with some learned parameters, and some hand-coded parameters that implement a deterministic
        path-finding algorithm."""
        super().__init__(cfg)
        self.oracle = FixedBfsNCA(cfg, requires_grad=True)
        self.oracle_out = None

    def forward(self, x):
        return super().forward(x)

    def forward_layer(self, x, i):
        # First, the underlying NCA processes the input as usual.
        x = super().forward_layer(x, i)

        # Then, feed the oracle some of the channels from the NCAs output.
        self.oracle_out = self.oracle.hid_forward(x[:, :self.oracle.n_in_chan + self.oracle.n_hid_chan])

        # print(x.shape)
        # Overwrite some hidden channels with the oracle's output. Note that this effectively makes the weights mapping
        # to these channels redundant. 
        # self.oracle_out = self.oracle.forward(x[:, -self.oracle.n_hid_chan:])
        # self.oracle_out = self.oracle.forward(self.oracle_out)
        y = th.cat([x[:, :-self.oracle_out.shape[1]], self.oracle_out], dim=1)
        # x[:, self.n_hid_chan - self.oracle.n_hid_chan: self.n_hid_chan, :, :] += self.oracle_out
        # x[:, self.n_hid_chan - self.oracle.n_hid_chan: self.n_hid_chan, :, :] = x[:, self.n_hid_chan - self.oracle.n_hid_chan: self.n_hid_chan, :, :] + self.oracle_out
        # print(f"Oracle requires_grad: {self.oracle_out.requires_grad}")
        # print(f"x requires_grad: {x.requires_grad}")

        return y
        # print(self.oracle_out.shape)
        # return self.oracle_out

    def reset(self, x0, is_torchinfo_dummy=False, **kwargs):
        with th.no_grad():
            self.oracle_out = th.zeros(x0.shape[0], self.oracle.n_hid_chan, x0.shape[2], x0.shape[3])
            # self.oracle_out[:, :self.oracle.n_in_chan] = x0

        # Don't give the oracle the maze directly off the top, we'll let the learned model feed it inputs at steps > 0.
        self.oracle.reset(self.oracle_out[:, :self.oracle.n_in_chan], is_torchinfo_dummy=is_torchinfo_dummy, **kwargs)
        # Or, feed the oracle the initial maze.
        # self.oracle.reset(x0, is_torchinfo_dummy=is_torchinfo_dummy, **kwargs)
        super().reset(x0, is_torchinfo_dummy)
