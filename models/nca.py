from collections import OrderedDict
from functools import partial
from pdb import set_trace as TT

import torch as th
from torch import nn
from configs.config import Config
from einops.layers.torch import Reduce

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
        max_pool = partial(MaxPools, cfg=cfg)
        

        def _make_conv():
            # This layer applies a dense layer to each 3x3 block of the input.
            conv = conv2d(
                n_hid_chan + n_in_chan, 
                # If we're not repeatedly feeding the input maze, replace this with some extra hidden channels.
                self.n_out_chan,
                kernel_size=cfg.kernel_size, 
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

        supp_modules = ()
        if cfg.max_pool:
            supp_modules = (max_pool,)

        if not cfg.shared_weights:
            modules = [nn.Sequential(OrderedDict([
                (f'conv_{i}', _make_conv()), 
                *[(f'supp_{i}', s()) for i, s in enumerate(supp_modules)],
                (f'relu_{i}', nn.ReLU())])) for i in range(cfg.n_layers)
            ]
        else:
            conv_0 = _make_conv()
            modules = [nn.Sequential(conv_0, *(s() for s in supp_modules), nn.ReLU()) for _ in range(cfg.n_layers)]

        self.layers = nn.ModuleList(modules)


    # def forward_layer(self, x: th.Tensor, i: int):
    #     ### DEBUG ###
    #     if self.n_step == 0:
    #         x[:] = 0.0
    #         # provide activation in top corner of first input
    #         x[0,-1, 0, 0] = 1.0
    #     else:
    #         # overwrite the maze
    #         x[:, :self.cfg.n_in_chan] = 0
        
    #     return super().forward_layer(x, i)

    # def reset(self, *args, **kwargs):
    #     ret = super().reset(*args, **kwargs)
    #     ### DEBUG ###
    #     for name, p in self.named_parameters():
    #         if "weight" in name:
    #             p.data.fill_(1.0)
    #         else:
    #             p.data.zero_()

    #     return ret


class MaxPools(nn.Module):
    """Take the max over the entire 2D input, outputting a scalar."""
    def __init__(self, cfg):
        super().__init__()
        # path_chan is where we'll be expecting the model to output the target path.
        self.spatial_pool_chan = cfg.path_chan + 1
        self.chan_pool_chan = cfg.path_chan + 2
        # self.map_pool_layer = nn.MaxPool2d((cfg.width, cfg.height))
        self.chan_pool_layer = Reduce('b c h w -> b 1 h w', 'max')

    def forward(self, x):
        map_kernel_size = x.shape[-2:]
        y_spatial = th.max_pool2d(x[:, self.spatial_pool_chan].clone(), map_kernel_size)
        # y_spatial = nn.MaxPool2d(map_kernel_size)(x[:, self.spatial_pool_chan].clone())
        x[:, self.spatial_pool_chan] += y_spatial
        # chan_kernel_size = (x.shape[1], 1, 1)
        # y_chan = th.max_pool3d(x, chan_kernel_size)
        y_chan = self.chan_pool_layer(x.clone())
        x[:, self.chan_pool_chan: self.chan_pool_chan + 1] = x[:, self.chan_pool_chan: self.chan_pool_chan + 1] + y_chan

        return x