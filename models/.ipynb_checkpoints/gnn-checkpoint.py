from collections import OrderedDict
from pdb import set_trace as TT
from turtle import right

import numpy as np
import torch as th
from torch import nn, Tensor
from torch_geometric.nn import GCNConv

from models.nn import PathfindingNN


class GCN(PathfindingNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        n_in_chan, n_hid_chan = cfg.n_in_chan, cfg.n_hid_chan
        self.grid_edges = None
        self.self_edges = None
        self.edges = None

        def _make_convs():
            gconv0 = GCNConv(n_hid_chan + n_in_chan, self.n_out_chan)
#             gconv0 = GCNConv(n_hid_chan + n_in_chan, n_hid_chan)
#             gconv1 = GCNConv(n_hid_chan, self.n_out_chan)


            # DEBUG
            # for p in self.parameters():
            #     p[:] = 0.0
            # with th.no_grad():
            #     lin_weight = list(gconv0.modules())[1].weight
            #     lin_weight = nn.Parameter(th.rand_like(lin_weight), requires_grad=False)
            #     list(gconv0.modules())[1].weight = lin_weight
            #     gconv0.bias = nn.Parameter(th.zeros_like(gconv0.bias), requires_grad=False)


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

        ### DEBUG ###
        # if self.n_step == 0:
        #     x[:] = 0.0
        #     x[0,-1, 0, 0] = 1.0

        batch_size, n_chan, width, height = x.shape

        # Move channel dimension to front. Then, flatten along width, height, and then batch dimensions.
        x = x.transpose(1, 0)
        x = x.reshape(x.shape[0], -1)
        x = x.transpose(1, 0)

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

    def reset(self, x0, is_torchinfo_dummy=False, new_batch_size=False):
        if self.edges is None or new_batch_size:
            batch_size = x0.shape[0]
            width, height = x0.shape[-2:]
            n_nodes = width * height
            grid_edges = get_grid_edges(width, height)
            self_edges = get_self_edges(width, height)
            self.grid_edges = batch_edges(grid_edges, batch_size, n_nodes)
            self.self_edges = batch_edges(self_edges, batch_size, n_nodes)
            self.edges = th.hstack((self.self_edges, self.grid_edges))
            # self.edges = self.self_edges
            # self.edges = self.grid_edges
        super().reset(x0, is_torchinfo_dummy)         


def get_grid_edges(width, height):
    """Create an edge matrix of shape (2, n_edges) for a graph corresponding to a 2D grid, in which grid cells sharing
    a border are connected by an edge."""
    # The 2D indices of all nodes, with shape (width, height, 2)
    node_indices = np.indices((width, height)).transpose(1, 2, 0)

    # The 2D indices of adjacent nodes in each of four directions, missing rows/columns corresponding to relevant 
    # borders, with shape, e.g., (width, height - 1, 2).
    left_indices, down_indices, right_indices, up_indices = \
        node_indices[:, :-1], node_indices[:-1], node_indices[:, 1:], node_indices[1:]

    # Arrays of edges in each direction, shape, e.g. (2, width, (height - 1), 2).
    left_edges, down_edges, right_edges, up_edges = \
        np.stack((node_indices[:, 1:], left_indices)), np.stack((node_indices[1:], down_indices)), \
        np.stack((node_indices[:, :-1], right_indices)), np.stack((node_indices[:-1], up_indices))

    # Arrays of edges, shape, e.g., (2, 2, width, (height - 1)) ~ (xy_coords, endpoints, nodes_per_row, nodes_per_col)
    dir_edges_lst = [e.transpose(3, 0, 1, 2) for e in [left_edges, down_edges, right_edges, up_edges]]

    # Array of flattened 2D indices, shape, e.g., (2, 2, width * (height - 1)) ~ (xy_coords, endpoints, nodes).
    edges = np.concatenate([e.reshape(2, 2, -1) for e in dir_edges_lst], -1)

    # Array of flattened 1D indices in flattened grid, e.g., (2, width * (height - 1))
    edges = np.ravel_multi_index(edges, (width, height))

    return th.Tensor(edges).long()

def get_self_edges(width, height):
    node_indices = np.indices((width, height)).transpose(1, 2, 0)
    self_edges = np.stack((node_indices, node_indices))
    self_edges = self_edges.transpose(3, 0, 1, 2).reshape(2, 2, -1)
    self_edges = np.ravel_multi_index(self_edges, (width, height))

    return th.Tensor(self_edges).long()


def batch_edges(edges, batch_size, n_nodes):
    """
    Batch edges into connected components width identical connectivity.

    For each set of edges, increment all the node indices to refer to the next connected component. 

    Args:
        edges (th.Tensor): Tensor of shape (2, n_edges), corresponding to the graph
        batch_size (int): Number of connected components in the output.
        n_nodes (int): Number of nodes in each connected component in the output.
    """
    n_edges = edges.shape[-1]
    edges = th.tile(edges, (1, batch_size))
    for i in range(1, batch_size):
        edges[:, i * n_edges:] += n_nodes

    return edges
