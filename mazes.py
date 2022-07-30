from enum import Enum
import os
from pathlib import Path
from pdb import set_trace as TT
import pickle

import hydra
import networkx as nx
import numpy as np
import scipy
import torch as th
from tqdm import tqdm

from configs.config import BatchConfig, Config
from grid_to_graph import get_traversable_grid_edges


maze_data_fname = os.path.join(Path(__file__).parent, os.path.join("data", "maze_data"))
path_chan = 4


class Tiles:
    EMPTY = 0
    WALL = 1
    SRC = 2
    TRG = 3


def get_maze_name_suffix(cfg: Config):
    maze_name_suffix = '' if cfg.width == cfg.height == 16 else f'_{cfg.width}x{cfg.height}'
    return maze_name_suffix


def main_mazes(cfg: Config=None):
    """Generate random mazes for training/testing."""
    maze_fname = maze_data_fname + get_maze_name_suffix(cfg)
    train_fname = f"{maze_fname}_train.pk"
    val_fname = f"{maze_fname}_val.pk"
    test_fname = f"{maze_fname}_test.pk"
    if np.any([os.path.exists(fname) for fname in [train_fname, val_fname, test_fname]]):
        overwrite = input("File already exists. Overwrite? (y/n) ")
        if overwrite != 'y':
            return
        
        print('Overwriting existing dataset...')

    maze_data = Mazes(cfg)
    with open(train_fname, 'wb') as f:
        pickle.dump(maze_data, f)

    maze_data = Mazes(cfg)
    with open(val_fname, 'wb') as f:
        pickle.dump(maze_data, f)

    maze_data = Mazes(cfg)
    with open(test_fname, 'wb') as f:
        pickle.dump(maze_data, f)

    # TODO: Render a big grid of all the data.


def load_dataset(cfg: Config, test_only: bool = False):
    """Load the dataset of random mazes"""
    maze_fname = maze_data_fname + get_maze_name_suffix(cfg)
    train_fname = f"{maze_fname}_train.pk"
    val_fname = f"{maze_fname}_val.pk"
    test_fname = f"{maze_fname}_test.pk"

    if not test_only:
        with open(train_fname, 'rb') as f:
            maze_data_train: Mazes = pickle.load(f)
        maze_data_train.get_subset(cfg)

        with open(val_fname, 'rb') as f:
            maze_data_val: Mazes = pickle.load(f)
        maze_data_val.get_subset(cfg)

    with open(test_fname, 'rb') as f:
        maze_data_test: Mazes = pickle.load(f)
    maze_data_test.get_subset(cfg)

    if not test_only:
        return maze_data_train.to(cfg.device), maze_data_val.to(cfg.device), maze_data_test.to(cfg.device)

    else:
        return maze_data_test.to(cfg.device)


class Mazes():
    def __init__(self, cfg: Config, evo_mazes={}):
        if len(evo_mazes) == 0:
            width, height = cfg.width, cfg.height
            self.mazes_discrete, self.mazes_onehot, self.edges, self.target_paths, self.target_diameters = gen_rand_mazes(cfg.n_data, cfg)
            self.maze_ims = th.Tensor(np.array(
                [render_discrete(maze_discrete[None,], cfg)[0] for maze_discrete in self.mazes_discrete]
            ))
        else:
            for k, v in evo_mazes.items():
                self.__setattr__(k, v)
    
    def to(self, device):
        self.mazes_discrete, self.mazes_onehot, self.target_paths = self.mazes_discrete.to(device), \
            self.mazes_onehot.to(device), self.target_paths.to(device)
        return self

    def get_subset(self, cfg: Config):
        self.mazes_discrete, self.mazes_onehot = self.mazes_discrete[:cfg.n_data], \
            self.mazes_onehot[:cfg.n_data]
        if cfg.task == 'pathfinding':
            self.target_paths = self.target_paths[:cfg.n_data]
        elif cfg.task == 'diameter':
            self.target_paths = self.target_diameters[:cfg.n_data]

            # Remove source and targets.
            self.mazes_discrete = th.where(
                    (self.mazes_discrete == Tiles.SRC) | (self.mazes_discrete == Tiles.TRG), 
                Tiles.EMPTY, self.mazes_discrete)
            assert Tiles.EMPTY == 0 and Tiles.WALL == 1 and Tiles.SRC == 2
            self.mazes_onehot = self.mazes_onehot[:, :Tiles.SRC]
        else:
            raise Exception


def generate_random_maze(cfg):
    batch_size = 1
    width, height = cfg.width, cfg.height
    empty_chan, wall_chan, src_chan, trg_chan = Tiles.EMPTY, Tiles.WALL, Tiles.SRC, Tiles.TRG

    # Generate empty room with with borders.
    rand_maze_onehot = th.zeros((batch_size, 4, width + 2, height + 2), dtype=int)
    rand_maze_onehot[:, wall_chan, [0, -1], :] = 1
    rand_maze_onehot[:, wall_chan, :, [0, -1]] = 1

    # Randomly generate wall/empty tiles.
    rand_walls = th.randint(0, 2, (batch_size, 1, width, height))
    rand_maze_onehot[:, wall_chan: wall_chan + 1, 1: -1, 1: -1] = rand_walls
    rand_maze_onehot[:, empty_chan: empty_chan + 1, 1: -1, 1: -1] = (rand_walls == 0).int()

    # Randomly generate sources/targets.
    src_xs, trg_xs = th.randint(0, width, (2, batch_size)) + 1
    src_ys, trg_ys = th.randint(0, height, (2, batch_size)) + 1

    # Remove wall/empty tiles at location of source/target.
    rand_maze_onehot[th.arange(batch_size), empty_chan, src_xs, src_ys] = 0
    rand_maze_onehot[th.arange(batch_size), wall_chan, src_xs, src_ys] = 0
    rand_maze_onehot[th.arange(batch_size), empty_chan, trg_xs, trg_ys] = 0
    rand_maze_onehot[th.arange(batch_size), wall_chan, trg_xs, trg_ys] = 0

    # Add sources and targets.
    rand_maze_onehot[th.arange(batch_size), src_chan, src_xs, src_ys] = 1
    rand_maze_onehot[th.arange(batch_size), trg_chan, trg_xs, trg_ys] = 1

    return rand_maze_onehot


def render_discrete(arr, cfg):
    empty_chan, wall_chan, src_chan, trg_chan = Tiles.EMPTY, Tiles.WALL, Tiles.SRC, Tiles.TRG
    empty_color = th.Tensor([1.0, 1.0, 1.0])
    wall_color = th.Tensor([0.0, 0.0, 0.0])
    src_color = th.Tensor([1.0, 1.0, 0.0])
    trg_color = th.Tensor([0.0, 1.0, 0.0])
    path_color = th.Tensor([1.0, 0.0, 0.0])
    colors = {empty_chan: empty_color, wall_chan: wall_color, path_chan: path_color}
    colors.update({src_chan: src_color, trg_chan: trg_color, })
    batch_size, height, width = arr.shape
    im = th.zeros((batch_size, height, width, 3), dtype=th.float32)

    for chan, color in colors.items():
        idxs = th.where(arr == chan)
        im[idxs[0], idxs[1], idxs[2], :] = color

    im = im.cpu().numpy()

    return im


# rand_maze_onehot = generate_random_maze()
# rand_maze_im = render_discrete(rand_maze_onehot.argmax(dim=1))
# fig, ax = plt.subplots()
# ax.imshow(np.hstack(rand_maze_im), )
# plt.tight_layout()

adj_coords_2d = np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
])


def get_graph(arr, passable=0, impassable=1):
    graph = nx.Graph()
    width, height = arr.shape
    size = width * height
    graph.add_nodes_from(range(size))
    # ret = scipy.sparse.csgraph.floyd_warshall(dist)
    for u in range(size):
        ux, uy = u // width, u % width
        if arr[ux, uy] ==impassable:
            continue
        neighbs_xy = [(ux - 1, uy), (ux, uy-1), (ux+1, uy), (ux, uy+1)]
        neighbs = [x * width + y for x, y in neighbs_xy]
        for v, (vx, vy) in zip(neighbs, neighbs_xy):
            if not 0 <= v < size or arr[vx, vy] == impassable:
                continue
            graph.add_edge(u, v)
    return graph


def bfs_grid(arr, passable=0, impassable=1, src=2, trg=3):
    srcs = np.argwhere(arr == src)
    assert srcs.shape[0] == 1
    src = tuple(srcs[0])
    return bfs(arr, src, trg, passable, impassable)


def bfs(arr, src: tuple[int], trg: tuple[int], passable=0, impassable=1):
    width = arr.shape[0]
    assert width == arr.shape[1]
    frontier = [src]
    back_paths = {}
    visited = set({})
    while frontier:
        curr = frontier.pop(0)
        if arr[curr[0], curr[1]] == trg:
            path = []
            path.append(curr)
            while curr in back_paths:
                curr = back_paths[curr]
                path.append(curr)
            return path[::-1]
        visited.add(curr)
        adjs = [tuple((np.array(curr) + move) % width) for move in adj_coords_2d]
        for adj in adjs:
            if adj in visited or arr[adj[0], adj[1]] == impassable:
                continue
            frontier.append(adj)
            back_paths.update({adj: curr})
    return []


def diameter(arr, passable=0, impassable=1):
    width, height = arr.shape
    graph = get_graph(arr, passable, impassable)
    shortest_paths = dict(nx.all_pairs_shortest_path(graph))
    max_connected = []
    max_path = []
    for u, connected in shortest_paths.items():
        if len(connected) > len(max_connected):
            max_connected = connected
        for v, path in connected.items():
            if len(path) > len(max_path):
                max_path = path
    max_connected_xy = [(u // width, u % width) for u in max_connected]
    max_path_xy = [(u // width, u % width) for u in max_path]
    return max_path_xy, max_connected_xy


def get_rand_path(arr, passable=0, impassable=1):
    width, height = arr.shape
    graph = get_graph(arr, passable, impassable)
    srcs = th.argwhere(arr==passable)
    src = srcs[np.random.randint(len(srcs))]
    src = src[0] * width + src[1]
    src = src.item()
    paths = dict(nx.shortest_path(graph, src))
    trg, path = paths.popitem()
    path_xy = [(i // width, i % width) for i in path]
    return path_xy


def gen_rand_mazes(n_data, cfg):
    # Generate new random mazes until we get enough solvable ones.
    n_data = cfg.n_data
    solvable_mazes_onehot = []
    solvable_mazes_discrete = []
    solvable_mazes_edges = []
    target_paths = []
    target_diameters = []
    i = 0
    j = 0
    for i in tqdm(range(n_data)):
        sol = None
        # Generate random mazes until we get a solvable one.
        while not sol:
            rand_maze_onehot = generate_random_maze(cfg)
            rand_maze_discrete = rand_maze_onehot.argmax(axis=1)
            sol = bfs_grid(rand_maze_discrete[0].cpu().numpy())
            j += 1
        # print(f'Adding maze {i}.')
        solvable_mazes_onehot.append(rand_maze_onehot)
        solvable_mazes_discrete.append(rand_maze_discrete)
        solvable_mazes_edges.append(get_traversable_grid_edges(rand_maze_onehot[0, Tiles.WALL] == 0))
        target_path = th.zeros_like(rand_maze_discrete)
        for x, y in sol:
                target_path[0, x, y] = 1
        target_paths.append(target_path)
        # For convenience, we use the same maze in the dataset for the diameter problem.
        target_diameter = th.zeros_like(rand_maze_discrete)
        diam, connected = diameter(rand_maze_discrete[0].cpu().numpy())
        for x, y in diam:
            target_diameter[0, x, y] = 1
        target_diameters.append(target_diameter)
        i += 1
    # print(f'Solution length: {len(sol)}') 
    print(f'Generated {j} random mazes to produce {n_data} solvable mazes.')

    return th.vstack(solvable_mazes_discrete), th.vstack(solvable_mazes_onehot), th.vstack(solvable_mazes_edges), \
        th.vstack(target_paths), th.vstack(target_diameters)


def get_target_path(maze_discrete, cfg):
    if cfg.task == 'pathfinding':
        # TODO: need to constrain mutation of sources and targets for this task.
        sol = get_rand_path(maze_discrete)
        x, y = sol[0]
        maze_discrete[x, y] = Tiles.SRC
        # for (x, y) in sol:
            # offspring_target_paths[mi, x, y] = 1
        maze_discrete[x, y] = Tiles.TRG
        return sol


def get_target_diam(maze_discrete, cfg):
    diam, connected = diameter(maze_discrete.cpu().numpy(), cfg.n_in_chan)
    # for (x, y) in diam:
        # offspring_target_paths[mi, x, y] = 1
    return diam


if __name__ == "__main__":
    main_mazes()
