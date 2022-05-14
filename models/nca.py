from collections import OrderedDict
from pdb import set_trace as TT

import torch as th
from torch import nn
from configs.config import Config

from models.nn import PathfindingNN


class NCA(PathfindingNN):
    def __init__(self, cfg: Config):
        """A Neural Cellular Automata model for pathfinding over grid-based mazes.
        
        Args:
            n_in_chan: Number of input channels in the onehot-encoded input.
            n_hid_chan: Number of channels in the hidden layers.
            drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
            """
        super().__init__(cfg)
        n_in_chan, n_hid_chan = cfg.n_in_chan, cfg.n_hid_chan
        # Number of hidden channels, also number of writable channels the the output. (None-maze channels at the input.)
        self.n_hid_chan = n_hid_chan    

        conv2d = nn.Conv2d


        def _make_conv():
            # This layer applies a dense layer to each 3x3 block of the input.
            conv = conv2d(
                n_hid_chan + n_in_chan, 
                # If we're not repeatedly feeding the input maze, replace this with some extra hidden channels.
                self.n_out_chan,
                kernel_size=3, 
                padding=1
            )
            if cfg.cut_conv_corners:
                conv.weight.data[:, :, 0, 0] = conv.weight.data[:, :, -1, -1] = conv.weight.data[:, :, 0, -1] = \
                    conv.weight.data[:, :, -1, 0] = 0

            if cfg.symmetric_conv:
                assert cfg.cut_conv_corners
                conv.weight.data[:, :, 1, 0] = conv.weight.data[:, :, 0, 1] = conv.weight.data[:, :, 1, 2] = \
                    conv.weight.data[:, :, 2, 1]

            return conv

        if not cfg.shared_weights:
            modules = [nn.Sequential(OrderedDict([(f'conv_{i}', _make_conv()), (f'relu_{i}', nn.ReLU())])) for i in range(cfg.n_layers)]
        else:
            conv_0 = _make_conv()
            modules = [nn.Sequential(conv_0, nn.ReLU()) for _ in range(cfg.n_layers)]

        self.layers = nn.ModuleList(modules)

