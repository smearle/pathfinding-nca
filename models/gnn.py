from pdb import set_trace as TT
from turtle import right

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid

from models.nn import PathfindingNN


class GCN(PathfindingNN):
    def __init__(self, n_in_chan, n_hid_chan, **kwargs):
        super().__init__()
        self.gconv1 = GCNConv(n_hid_chan + n_in_chan, n_hid_chan)
        self.gconv2 = GCNConv(n_hid_chan, n_hid_chan)
        self.grid_edges = None
        self.self_edges = None

    def forward(self, x: Tensor) -> Tensor:
        """Take in a batched 2D maze, then preprocess for consumption by a graph neural network.
        
        This involves concatenating with the underlying maze-state (in `super.forward()`), and flattening along the 
        maze's width and height dimensions."""
        batch_size, n_chan, width, height = x.shape

        # Cannot use batches with pyg (apparently...?)
        assert batch_size == 1

        x = super().forward(x)

        # Remove batch dimension
        x = x[0]

        if self.grid_edges is None:
            self.grid_edges = get_grid_edges(*x.shape[-2:])  # [None,...]
            self.self_edges = get_self_edges(*x.shape[-2:])

        # Flatten along width and height dimensions and remove batch dimension.
        x = x.view(x.shape[0], -1)

        # Batch the edge indices
        # edge_index = self.grid_edges * torch.ones((x.shape[0], *self.grid_edges.shape[1:]))

        x = self.forward_gnn(x, self.grid_edges)

        # Reshape back into (bathched) 2D grid.
        x = x.reshape(batch_size, n_chan, width, height)

        return x

    def forward_gnn(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = x.transpose(1, 0)
        x = self.gconv1(x, edge_index).relu()
        x = self.gconv2(x, self.self_edges)
        return x


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

    return torch.Tensor(edges).long()

def get_self_edges(width, height):
    node_indices = np.indices((width, height)).transpose(1, 2, 0)
    self_edges = np.stack((node_indices, node_indices))
    self_edges = self_edges.transpose(3, 0, 1, 2).reshape(2, 2, -1)
    self_edges = np.ravel_multi_index(self_edges, (width, height))

    return torch.Tensor(self_edges).long()


