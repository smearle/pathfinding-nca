
from collections import OrderedDict
from pdb import set_trace as TT

import torch as th
from torch import nn
from configs.config import Config

from models.fixed_nca import FixedBfsNCA
from models.nca import NCA
from models.nn import PathfindingNN


class BfsNCA(NCA):
    def __init__(self, cfg: Config):
        """A hybrid model with some learned parameters, and some hand-coded parameters that implement a deterministic
        path-finding algorithm."""
        super().__init__(cfg)
        self.oracle = FixedBfsNCA(cfg)
        self.oracle_out = None

    def forward_layer(self, x, i):
        self.oracle_out = self.oracle(self.oracle_out)
        # Overwrite some hidden channels with the oracle's output. Note that this effectively makes the weights mapping
        # to these channels redundant. 
        x[:, self.n_hid_chan - self.oracle.n_hid_chan: self.n_hid_chan, :, :] = self.oracle_out

        return super().forward_layer(x, i)

    def reset(self, x0, is_torchinfo_dummy=False, **kwargs):
        self.oracle_out = th.zeros(x0.shape[0], self.oracle.n_hid_chan, x0.shape[2], x0.shape[3])
        self.oracle.reset(x0)
        super().reset(x0, is_torchinfo_dummy)
