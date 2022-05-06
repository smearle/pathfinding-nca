from pdb import set_trace as TT
from turtle import right

import numpy as np
import torch as th
from torch import nn
from torch import Tensor
from torch_geometric.nn import GCNConv

from models.nn import PathfindingNN


class GCN(PathfindingNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        n_in_chan, n_hid_chan = cfg.n_in_chan, cfg.n_hid_chan
        self.grid_edges = None
        self.self_edges = None

        def _make_convs():
            gconv0 = GCNConv(n_hid_chan + n_in_chan, n_hid_chan)
            gconv1 = GCNConv(n_hid_chan, n_hid_chan + (n_in_chan if not cfg.skip_connections else 0))
            return gconv0, gconv1
            
        if not cfg.shared_weights:
            layers = [_make_convs() for _ in range(cfg.n_layers)]
        else:
            gconv0, gconv1 = _make_convs()
            layers = [(gconv0, gconv1)] * cfg.n_layers
        
        self.layers = nn.ModuleList([l for lt in layers for l in lt])

    def forward_layer(self, x: Tensor, i: int) -> Tensor:
        """Take in a batched 2D maze, then preprocess for consumption by a graph neural network.
        
        This involves concatenating with the underlying maze-state (in `super.forward()`), and flattening along the 
        maze's width and height dimensions."""
        batch_size, n_chan, width, height = x.shape

        # Cannot use batches with pyg (apparently...?)
        # assert batch_size == 1

        # Remove batch dimension, because GNNs don't allow batches (??) :'(
        # x = x[0]

        # Flatten along width, height, and then batch dimensions.
        x = x.view(x.shape[1], -1)

        x = x.transpose(1, 0)
        x = self.layers[i*2](x, self.grid_edges).relu()
        x = self.layers[i*2+1](x, self.self_edges).relu()

        # Batch the edge indices
        # edge_index = self.grid_edges * th.ones((x.shape[0], *self.grid_edges.shape[1:]))

        # Reshape back into (batched) 2D grid.
        x = x.reshape(batch_size, n_chan, width, height)

        return x

    def reset(self, x0, is_torchinfo_dummy=False):
        if self.grid_edges is None:
            self.grid_edges = get_grid_edges(x0.shape[0], *x0.shape[-2:])
            self.self_edges = get_self_edges(x0.shape[0], *x0.shape[-2:])
        super().reset(x0, is_torchinfo_dummy)         


def get_grid_edges(batch_size, width, height):
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

    edges = batch_edges(batch_size, edges)

    return th.Tensor(edges).long()

def get_self_edges(batch_size, width, height):
    node_indices = np.indices((width, height)).transpose(1, 2, 0)
    self_edges = np.stack((node_indices, node_indices))
    self_edges = self_edges.transpose(3, 0, 1, 2).reshape(2, 2, -1)
    self_edges = np.ravel_multi_index(self_edges, (width, height))

    edges = batch_edges(batch_size, self_edges)

    return th.Tensor(self_edges).long()


def batch_edges(batch_size, edges):
    # Batch it
    n_edges = edges.shape[1]
    edges = np.tile(edges, (1, batch_size))
    for i in range(batch_size):
        ii = i * n_edges
        edges[ii:] += ii

    return edges
