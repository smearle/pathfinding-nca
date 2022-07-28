
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
        x = super().forward_layer(x, i)

        # Overwrite some hidden channels with the oracle's output. Note that this effectively makes the weights mapping
        # to these channels redundant. 
        # with th.no_grad():
        self.oracle_out = self.oracle(self.oracle_out)
        x[:, self.n_hid_chan - self.oracle.n_hid_chan: self.n_hid_chan, :, :] += self.oracle_out

        return x

    def reset(self, x0, is_torchinfo_dummy=False, **kwargs):
        # with th.no_grad():
        self.oracle_out = th.zeros(x0.shape[0], self.oracle.n_hid_chan, x0.shape[2], x0.shape[3])
        self.oracle.reset(x0)
        super().reset(x0, is_torchinfo_dummy)
