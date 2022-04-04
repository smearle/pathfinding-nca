from pdb import set_trace as TT

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='.', name='Cora')

# TODO: Generate the edge index matrix for mazes.
TT()

class GCN(torch.nn.Module):
    def __init__(self, n_in_chan, n_hid_chan):
        super().__init__()
        self.conv1 = GCNConv(n_in_chan + n_hid_chan, n_hid_chan)
        self.conv2 = GCNConv(n_hid_chan, n_hid_chan)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.forward_gnn(x, x.new_empty(0))
        return x

    def forward_gnn(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
