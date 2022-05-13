import os
from pathlib import Path
from pdb import set_trace as TT
import pickle

import networkx as nx
import numpy as np
import scipy
import torch as th
from tqdm import tqdm

from configs.config import Config


path_chan = 4
maze_data_fname = os.path.join(Path(__file__).parent, os.path.join("data", "maze_data"))
train_fname = f"{maze_data_fname}_train.pk"
val_fname = f"{maze_data_fname}_val.pk"
test_fname = f"{maze_data_fname}_test.pk"


def main_mazes(cfg: Config):
    """Generate random mazes for training/testing."""
    if np.any([os.path.exists(fname) for fname in [train_fname, val_fname, test_fname]]):
        overwrite = input("File already exists. Overwrite? (y/n) ")
        if overwrite != 'y':
            return
        
        print('Overwriting existing dataset...')

    maze_data = Mazes(cfg.n_data, cfg)
    with open(train_fname, 'wb') as f:
        pickle.dump(maze_data, f)

    maze_data = Mazes(cfg.n_val_data, cfg)
    with open(val_fname, 'wb') as f:
        pickle.dump(maze_data, f)

    maze_data = Mazes(cfg.n_test_data, cfg)
    with open(test_fname, 'wb') as f:
        pickle.dump(maze_data, f)

    # TODO: Render a big grid of all the data.


def load_dataset(cfg: Config):
    """Load the dataset of random mazes"""
    with open(train_fname, 'rb') as f:
        maze_data_train: Mazes = pickle.load(f)
    maze_data_train.get_subset(cfg)

    with open(val_fname, 'rb') as f:
        maze_data_val: Mazes = pickle.load(f)
    maze_data_val.get_subset(cfg)

    with open(test_fname, 'rb') as f:
        maze_data_test: Mazes = pickle.load(f)
    maze_data_test.get_subset(cfg)

    return maze_data_train.to(cfg.device), maze_data_val.to(cfg.device), maze_data_test.to(cfg.device)


class Mazes():
    def __init__(self, cfg: Config, evo_mazes={}):
        if len(evo_mazes) == 0:
            width, height = cfg.width, cfg.height
            self.mazes_discrete, self.mazes_onehot, self.target_paths, self.target_diameters = gen_rand_mazes(cfg.n_data, cfg)
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
                    (self.mazes_discrete == cfg.src_chan) | (self.mazes_discrete == cfg.trg_chan), 
                cfg.empty_chan, self.mazes_discrete)
            assert cfg.empty_chan == 0 and cfg.wall_chan == 1 and cfg.src_chan == 2
            self.mazes_onehot = self.mazes_onehot[:, :cfg.src_chan]
        else:
            raise Exception
        


def generate_random_maze(cfg):
    batch_size = 1
    width = 16
    empty_chan, wall_chan, src_chan, trg_chan = cfg.empty_chan, cfg.wall_chan, cfg.src_chan, cfg.trg_chan

    # Generate empty room with with borders.
    rand_maze_onehot = th.zeros((batch_size, 4, width + 2, width + 2), dtype=int)
    rand_maze_onehot[:, wall_chan, [0, -1], :] = 1
    rand_maze_onehot[:, wall_chan, :, [0, -1]] = 1

    # Randomly generate wall/empty tiles.
    rand_walls = th.randint(0, 2, (batch_size, 1, width, width))
    rand_maze_onehot[:, wall_chan: wall_chan + 1, 1: -1, 1: -1] = rand_walls
    rand_maze_onehot[:, empty_chan: empty_chan + 1, 1: -1, 1: -1] = (rand_walls == 0).int()

    # Randomly generate sources/targets.
    src_xs, src_ys, trg_xs, trg_ys = th.randint(0, width, (4, batch_size)) + 1

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
    empty_chan, wall_chan, src_chan, trg_chan = cfg.empty_chan, cfg.wall_chan, cfg.src_chan, cfg.trg_chan
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


# Algorithm 
def floyd_2d(arr, passable=0, impassable=1):
    width, height = arr.shape
    size = width * height
    dist = np.empty((size, size))
    dist.fill(np.inf)
    # ret = scipy.sparse.csgraph.floyd_warshall(dist)
    paths = [[[] for _ in range(size)] for _ in range(size)]
    for i in range(size):
        x, y = i // width, i % width
        neighbs = [(x - 1, y), (x, y-1), (x+1, y), (x, y+1)]
        neighbs = [x * width + y for x, y in neighbs]
        for j in neighbs:
            if not 0 <= j < size:
                continue
            dist[i, j] = 1
            paths[i][j] = [j]

    # Adding vertices individually
    for r in range(size):
        for p in range(size):
            for q in range(size):
                dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])
                if dist[p][q] < dist[p][r] + dist[r][q]:
                    continue
                else:
                    paths[p][q] = paths[p][r] + paths[r][q]

    src = np.argwhere(dist)
    raise Exception


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


def gen_rand_mazes(n_data, cfg):
    # Generate new random mazes until we get enough solvable ones.
    n_data = cfg.n_data
    solvable_mazes_onehot = []
    solvable_mazes_discrete = []
    target_paths = []
    target_diameters = []
    i = 0
    for i in tqdm(range(n_data)):
    # while len(solvable_mazes_onehot) < n_data:
        sol = None
        while not sol:
            rand_maze_onehot = generate_random_maze(cfg)
            rand_maze_discrete = rand_maze_onehot.argmax(axis=1)
            sol = bfs_grid(rand_maze_discrete[0].cpu().numpy())
        # print(f'Adding maze {i}.')
        solvable_mazes_onehot.append(rand_maze_onehot)
        solvable_mazes_discrete.append(rand_maze_discrete)
        target_path = th.zeros_like(rand_maze_discrete)
        for x, y in sol:
                target_path[0, x, y] = 1
        target_paths.append(target_path)
        target_diameter = th.zeros_like(rand_maze_discrete)
        diam, connected = diameter(rand_maze_discrete[0].cpu().numpy())
        for x, y in diam:
            target_diameter[0, x, y] = 1
        target_diameters.append(target_diameter)
        i += 1
    # print(f'Solution length: {len(sol)}')
    print(f'Generated {i} random mazes to produce {n_data} solvable mazes.')

    return th.vstack(solvable_mazes_discrete), th.vstack(solvable_mazes_onehot), th.vstack(target_paths), \
        th.vstack(target_diameters)


if __name__ == "__main__":
    # Number of vertices
    nV = 4
    INF = 999

    cfg = Config()
    main_mazes(cfg)
