from collections import OrderedDict
from functools import partial
import math
from pdb import set_trace as TT
from turtle import right
from typing import Iterable
from einops import rearrange

import numpy as np
import torch as th
from torch import nn, Tensor
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from configs.config import Config
from grid_to_graph import get_neighb_edges, get_self_edges
from mazes import Tiles

from models.nn import PathfindingNN


np.set_printoptions(threshold=np.inf, linewidth=np.inf)
th.set_printoptions(threshold=1000, linewidth=1000)


class GNN(PathfindingNN):
    """Parent class for graph neural network models."""
    def __init__(self, cfg, conv: MessagePassing):
        super().__init__(cfg)
        n_in_chan, n_hid_chan = cfg.n_in_chan, cfg.n_hid_chan
        self.neighb_edges = None
        self.self_edges = None
        self.edges = None

        def _make_convs():
            gconv0 = conv(
                in_channels=n_hid_chan + n_in_chan, 
                out_channels=self.n_out_chan, 
                add_self_loops=False, 
            )
            weight = list(gconv0.modules())[1].weight
            bias = gconv0.bias

            th.nn.init.normal_(weight, 0, 0.01)
            th.nn.init.normal_(bias, 0, 0.01)

            # _dummy_conv = th.nn.Conv2d(n_hid_chan + n_in_chan, self.n_out_chan, 3, 1, 1)
            # weight.data[:] = _dummy_conv.weight[:,:,1,0]
            # bias.data[:] = _dummy_conv.bias[:]

            # NOTE: copy-pasted from torch Conv2d implementation. To match the initialization scheme for a 3x3 NCA with
            #   the same weights, we get a bit more hands-on.
            # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
            # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
            # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
            # th.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

            # # Here is the conv weight we're attempting to match.
            # trg_kernel_shape = (3, 3)
            # trg_kernel_size = math.prod(trg_kernel_shape)
            # # Number of channels is the same as in a conv weight.
            # k = weight.size(1) * trg_kernel_size
            # th.nn.init.uniform_(weight, -1/math.sqrt(k), 1/math.sqrt(k))
            # assert bias is not None
            # fan_in, _ = th.nn.init._calculate_fan_in_and_fan_out(weight)
            # # Not sure of specifics of `fan_in` but this seems to match the outcome in the case of a conv weight.
            # fan_in *= trg_kernel_size
            # if fan_in != 0:
            #     bound = 1 / math.sqrt(fan_in)
            #     th.nn.init.uniform_(bias, -bound, bound)

            return gconv0  #, gconv1
            
        if not cfg.shared_weights:
            layers = [_make_convs() for _ in range(cfg.n_layers)]
        else:
            gconv0 = _make_convs()
            layers = [gconv0] * cfg.n_layers
#             gconv0, gconv1 = _make_convs()
#             layers = [(gconv0, gconv1)] * cfg.n_layers
        
#         self.layers = nn.ModuleList([l for lt in layers for l in lt])
        self.layers = nn.ModuleList(layers)

    def forward_layer(self, x: Tensor, i: int) -> Tensor:
        """Take in a batched 2D maze, then preprocess for consumption by a graph neural network.
        
        This involves concatenating with the underlying maze-state (in `super.forward()`), and flattening along the 
        maze's width and height dimensions."""

        # ### DEBUG ###
        if self.n_step == 0:
            x[:] = 0.0
            # provide activation in top corner of first input
            x[0,-1, 0, 0] = 1.0
        else:
            # overwrite the maze
            x[:, :self.cfg.n_in_chan] = 0
        ### DEBUG ###

        batch_size, n_chan, width, height = x.shape

        # Move channel dimension to front. Then, flatten along height, width, and then batch dimensions.
        x = x.transpose(1, 0)
        x = x.reshape(x.shape[0], -1)

        # (batches, channels)
        x = x.transpose(1, 0)

        # x = self.layers[i](x, self.grid_edges).relu()
        # x = th.clip(self.layers[i](x, self.edges), 0, 1)
        x = self.layers[i](x, self.edges).relu()

#         x = self.layers[i*2](x, self.grid_edges).relu()
#         x = self.layers[i*2+1](x, self.self_edges).relu()

        # Batch the edge indices
        # edge_index = self.grid_edges * th.ones((x.shape[0], *self.grid_edges.shape[1:]))

        # Reshape back into (batched) 2D grid.
        x = x.transpose(1, 0)
        x = x.reshape(self.n_out_chan, batch_size, width, height)
        x = x.transpose(1, 0)

        return x

    def reset(self, x0, e0: th.Tensor = None, is_torchinfo_dummy=False, new_batch_size=False):
        """Input the initial 2D grid maze to the network.

        Args:
            x0 (th.Tensor): A batch of mazes. Shape: (batch_size, n_in_chan, width, height).
            e0 (th.Tensor, optional): A batch of maze edges. Shape: (batch_size, 2, n_edges). When present, we assume the 2D grid 
                maze has been translated to a graph by representing traversible tiles as nodes, with edges between adjacent 
                traversible tiles. Otherwise all tiles are nodes, with edges to all adjacent tiles. Defaults to None.
            is_torchinfo_dummy (bool, optional): If we are receiving empty tensors to retrieve architecture info. 
                Defaults to False.
            new_batch_size (bool, optional): If the batch size has changed relative to previous episodes, in which case
                we may need to re-initialize the edge matrix. Defaults to False.
        """
        if e0 is None:
            if (self.edges is None or new_batch_size):
                batch_size = x0.shape[0]
                width, height = x0.shape[-2:]
                n_nodes = width * height
                grid_edges = get_neighb_edges(width, height)
                self_edges = get_self_edges(width, height)
                self.neighb_edges = batch_edges(th.tile(grid_edges, (1, batch_size)), [n_nodes] * batch_size)
                self.self_edges = batch_edges(th.tile(self_edges, (1, batch_size)), [n_nodes] * batch_size)
                self.edges = th.hstack((self.self_edges, self.neighb_edges))
                # self.edges = self.self_edges
                # self.edges = self.grid_edges
        else:
            # NOTE: We assume that during each episode (i.e. between calls to reset) the model is exposed only to *the
            #  same* batch of mazes. Otherwise `self.edges` will be incorrect (as would `self.initial_maze`, but it 
            #  bears repeating).
            # TODO:...
            # Modify e0, incrementing each set of edges by then number of nodes in all previous mazes.
            self.edges = batch_edges(edges=e0, n_nodes=th.sum(x0[:, Tiles.WALL, ...] != 1, axis=0))
            pass
        super().reset(x0, is_torchinfo_dummy)         


        # ### DEBUG ###
        for name, p in self.named_parameters():
            if "weight" in name:
                p.data.fill_(1.0)
            else:
                p.data.zero_()


class GCN(GNN):
    """Graph convolutional network, which applies the same MLP to each edge of a node, then aggregates the results."""
    def __init__(self, cfg: Config):
        conv = partial(GCNConv, improved=False, normalize=False)
        super().__init__(cfg, conv=conv)


class GAT(GNN):
    """A graph attention network, which #TODO..."""
    def __init__(self, cfg: Config):
        super().__init__(cfg, conv=GATConv)


def batch_edges(edges: Iterable[th.Tensor], n_nodes: Iterable[int]):
    """
    Batch edges into connected components width identical connectivity.

    For each set of edges, increment all the node indices to refer to the next connected component. 

    Args:
        edges (Iterable[th.Tensor]): List of tensors of shape (2, n_edges), corresponding to each input graph (connected 
            component in the output graph).
        n_nodes (Iterable[int]): Number of nodes in each input graph.
    """
    n_prev_nodes = 0
    for i, (edge_set, n_subgraph_nodes) in enumerate(zip(edges, n_nodes)):
        edge_set += n_prev_nodes
        edges[i] = edge_set
        n_prev_nodes += n_subgraph_nodes

    return edges
