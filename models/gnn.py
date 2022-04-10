from pdb import set_trace as TT

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid


# dataset = Planetoid(root='.', name='Cora')


class GCN(torch.nn.Module):
    def __init__(self, n_in_chan, n_hid_chan, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(n_in_chan + n_hid_chan, n_hid_chan)
        self.conv2 = GCNConv(n_hid_chan, n_hid_chan)
        self.maze_edges = None

    def forward(self, x: Tensor) -> Tensor:

        if self.maze_edges is None:
            width, height = x.shape[-2:]
            node_indices = np.indices(x.shape[-2:]).transpose(1, 2, 0)
            # up_indices = 
            # down_indices = 
            TT()
            maze_edges = [[(i, j), (i-1, j)] for i in range(1, width) for j in range(height)]
            maze_edges += [[(i, j), (i+1, j)] for i in range(width-1) for j in range(height)]
            maze_edges += [[(i, j), (i, j-1)] for i in range(width) for j in range(1, height)]
            maze_edges += [[(i, j), (i, j+1)] for i in range(width) for j in range(height-1)]
            self.maze_edges = maze_edges

        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.forward_gnn(x, self.maze_edges)
        return x

    def forward_gnn(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
