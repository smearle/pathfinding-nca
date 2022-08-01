from pdb import set_trace as TT

import numpy as np
import torch as th


def get_traversable_grid_edges(traversable: th.Tensor):
    """Convert a binary grid maze of traversable/non-traversable tiles into a graph in which nodes are traversable tiles
    and edges connect traversable tiles to adjacent traversable tiles.
    
    Args:
        traversable (th.Tensor[bool]): A 2D boolean array. True/False elements correspond to traversable/non-traversable
            tiles.
    """
    neighb_edges, neighb_edge_feats = get_neighb_edges(traversable.shape[0], traversable.shape[1], traversable)
    self_edges, self_edge_feats = get_self_edges(traversable.shape[0], traversable.shape[1], traversable)
    edges = th.hstack((neighb_edges, self_edges))
    edge_feats = th.vstack((neighb_edge_feats, self_edge_feats))
    return edges, edge_feats


def get_neighb_edges(width, height, traversable: th.Tensor = None):
    """Create an edge matrix of shape (2, n_edges) for a graph corresponding to a 2D grid, in which grid cells sharing
    a border are connected by an edge.
    
    Args:
        width (int): Width of the grid.
        height (int): Height of the grid.
        traversable (th.Tensor, optional): A tensor of shape (width, height) indicating whether each grid cell is 
            traversable. If None, we include each tile in the grid as a node in the graph. Otherwise, we include only 
            traversable tiles. Defaults to None.
    """
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

    edge_sets = [left_edges, down_edges, right_edges, up_edges]

    ### DEBUG ###
    # traversable = th.randn((width, height)) > 0
    ### DEBUG ###

    # Arrays of edges, shape, e.g., (2, 2, width, (height - 1)) ~ (xy_coords, endpoints, nodes_per_row, nodes_per_col)
    dir_edges_lst = [e.transpose(3, 0, 1, 2) for e in edge_sets]

    # Array of flattened 2D indices, shape, e.g., (2, 2, width * (height - 1)) ~ (xy_coords, endpoints, nodes).
    edges = np.concatenate([e.reshape(2, 2, -1) for e in dir_edges_lst], -1)

    edge_features = th.zeros((edges.shape[-1], 2))
    n = 0
    for n_edges, adj in [(height * (width - 1), [0, -1]), ((height - 1) * width, [1, 0]), \
            (height * (width - 1), [0, 1]), ((height - 1) * width, [-1, 0])]:
        edge_features[n: n + n_edges] = th.Tensor(adj)
        n += n_edges

    if traversable is not None:
        edges = filter_nontraversable_edges(edges, traversable)

    # Array of flattened 1D indices in flattened grid, e.g., (2, width * (height - 1))
    edges = np.ravel_multi_index(edges, (width, height))

    return th.Tensor(edges).long(), edge_features


def get_self_edges(width, height, traversable: th.Tensor = None):
    node_indices = np.indices((width, height)).transpose(1, 2, 0)
    self_edges = np.stack((node_indices, node_indices))
    self_edges = self_edges.transpose(3, 0, 1, 2).reshape(2, 2, -1)

    if traversable is not None:
        self_edges = filter_nontraversable_edges(self_edges, traversable)

    self_edges = np.ravel_multi_index(self_edges, (width, height))

    return th.Tensor(self_edges).long(), th.zeros((self_edges.shape[-1], 2))


def filter_nontraversable_edges(edges, traversable):
    """Filter out edges that include non-traversable tiles."""
    return edges[:, :, np.where(traversable[edges[0]] & traversable[edges[1]])[0]]
