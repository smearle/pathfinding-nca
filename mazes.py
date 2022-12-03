from enum import Enum
import os
from pathlib import Path
from pdb import set_trace as TT
import pickle
from typing import Iterable
import PIL
from einops import rearrange

import hydra
import networkx as nx
import numpy as np
import scipy
import torch as th
from tqdm import tqdm
from configs.config import Config

from grid_to_graph import get_traversable_grid_edges


# TODO: We should generate and store data for pathfinding, diameter, TSP separately, in case we want different types of
#  mazes for different tasks. (For now, we generate multi-hot mazes that share walls.)


maze_data_fname = os.path.join(Path(__file__).parent, os.path.join("data", "maze_data"))
path_chan = 4


class Tiles:
    # Base traversible.
    EMPTY = 0
    WALL = 1
    # Player start.
    SRC = 2
    # Exit.
    TRG = 3
    # TSP destinations.
    DEST = 4

    MAX = DEST

class Tilesets:

    class Tileset():
        # This determines rendering order
        tiles = []
        def __init__(self):
            self.tile_idxs = {tile: i for i, tile in enumerate(self.tiles)}

    class PATHFINDING(Tileset):
        tiles = [Tiles.EMPTY, Tiles.WALL, Tiles.SRC, Tiles.TRG]

    class DIAMETER(Tileset):
        tiles = [Tiles.EMPTY, Tiles.WALL]

    class TRAVELING(Tileset):
        tiles = [Tiles.EMPTY, Tiles.WALL, Tiles.DEST]

    MAZE_GEN = DIAMETER
    PATHFINDING_SOLNFREE = PATHFINDING


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
    #     user_input = input("File already exists. Overwrite? (y/n) ")
    #     while user_input not in ['y', 'n']:
    #         print(f"Invalid input '{user_input}'. Please enter 'y' or 'n'.")
    #         user_input = input("File already exists. Overwrite? (y/n) ")
    #     if user_input != 'y':
    #         return
        
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
            maze_data_train.to(cfg.device)
        maze_data_train.get_subset(cfg)

        with open(val_fname, 'rb') as f:
            maze_data_val: Mazes = pickle.load(f)
            maze_data_val.to(cfg.device)
        maze_data_val.get_subset(cfg)

    with open(test_fname, 'rb') as f:
        maze_data_test: Mazes = pickle.load(f)
        maze_data_test.to(cfg.device)
    maze_data_test.get_subset(cfg)

    if not test_only:
        return maze_data_train.to(cfg.device), maze_data_val.to(cfg.device), maze_data_test.to(cfg.device)

    else:
        return maze_data_test.to(cfg.device)


class Mazes():
    def __init__(self, cfg: Config, evo_mazes={}):
        if len(evo_mazes) == 0:
            width, height = cfg.width, cfg.height
            self.gen_rand_mazes(cfg.n_data, cfg)
            self.maze_ims = th.Tensor(np.array(
                # [render_discrete(maze_discrete[None,], cfg)[0] for maze_discrete in self.mazes_discrete]
                [render_multihot(maze_onehot[None,], cfg)[0] for maze_onehot in self.mazes_onehot]
            ))
        else:
            for k, v in evo_mazes.items():
                self.__setattr__(k, v)
    
    def to(self, device):
        self.mazes_discrete, self.mazes_onehot, self.target_paths = self.mazes_discrete.to(device), \
            self.mazes_onehot.to(device), self.target_paths.to(device)
        return self

    def gen_rand_mazes(self, n_data, cfg):
        # Generate new random mazes until we get enough solvable ones.
        n_data = cfg.n_data
        solvable_mazes_onehot = []
        solvable_mazes_discrete = []
        solvable_mazes_edges = []
        solvable_mazes_edge_feats = []
        target_paths = []
        target_diameters = []
        target_travelings = []
        i = 0
        j = 0
        for i in tqdm(range(n_data)):
            sol = None
            target_traveling = None
            # Generate random mazes until we get a solvable one.
            while not sol:
                rand_maze_onehot = generate_random_maze(cfg)
                rand_maze_discrete = rand_maze_onehot.argmax(axis=1)
                graph, edges, edge_feats, src, trg = get_graph(rand_maze_onehot[0])
                assert not src == trg
                # sol = bfs_grid(rand_maze_discrete[0].cpu().numpy())
                width = rand_maze_discrete.shape[1]
                sol = bfs_nx(width, graph, src, trg)
                j += 1

            # print(f'Adding maze {i}.')
            solvable_mazes_discrete.append(rand_maze_discrete)
            # solvable_mazes_edges.append(get_traversable_grid_edges(rand_maze_onehot[0, Tiles.WALL] == 0))
            solvable_mazes_edges.append(edges)
            solvable_mazes_edge_feats.append(edge_feats)
            target_path = th.zeros_like(rand_maze_discrete)
            sol = np.array(sol)
            target_path[0, sol[:, 0], sol[:, 1]] = 1
            # if not th.all(th.conv2d(weight=th.Tensor([[[[1,1],[1,1]]]]).float(), input=target_path.float(), bias=th.Tensor([-4])) < 0):
            #     TT()
            assert th.all(th.conv2d(weight=th.Tensor([[[[1,1],[1,1]]]]).float(), input=target_path.float(), bias=th.Tensor([-4])) < 0)
            target_paths.append(target_path)
            # For convenience, we use the same maze in the dataset for the diameter problem.
            diam_xy, connected, traveling_xy, dests_xy = diameter(width, graph, traveling=True)

            # Weirdly adding destinations outside the relevant function :)
            rand_maze_onehot[0, Tiles.DEST, dests_xy[:, 0], dests_xy[:, 1]] = 1
            # rand_maze_onehot[0, Tiles.WALL, dests_xy[:, 0], dests_xy[:, 1]] = 0 
            # rand_maze_onehot[0, Tiles.EMPTY, dests_xy[:, 0], dests_xy[:, 1]] = 0

            diam_xy = np.array(diam_xy)
            target_diameter = th.zeros_like(rand_maze_discrete)
            target_diameter[0, diam_xy[:, 0], diam_xy[:, 1]] = 1
            traveling_xy = np.array(traveling_xy)
            target_traveling = th.zeros_like(rand_maze_discrete)
            target_traveling[0, traveling_xy[:, 0], traveling_xy[:, 1]] = 1
            target_diameters.append(target_diameter)
            solvable_mazes_onehot.append(rand_maze_onehot)
            target_travelings.append(target_traveling)
            i += 1
        print(f'Generated {j} random mazes to produce {n_data} solvable mazes.')

        self.mazes_discrete, self.mazes_onehot, self.edges, self.edge_feats, self.target_paths, self.target_diameters, self.target_travelings = \
            th.vstack(solvable_mazes_discrete), th.vstack(solvable_mazes_onehot), solvable_mazes_edges, \
            solvable_mazes_edge_feats, th.vstack(target_paths), th.vstack(target_diameters), th.vstack(target_travelings)

    def get_tiles(self, tiles):
        """Need to give *all* tiles not to be left in maze in in_tiles."""
        mask = th.zeros(self.mazes_onehot.shape)
        mask[:, th.Tensor(tiles).long()] = 1
        mazes_onehot = self.mazes_onehot * mask
        self.mazes_discrete = th.argmax(mazes_onehot, axis=1)
        self.mazes_onehot = self.mazes_onehot[:, np.array(tiles)]


    def get_subset(self, cfg: Config):
        self.mazes_discrete, self.mazes_onehot = self.mazes_discrete[:cfg.n_data], \
            self.mazes_onehot[:cfg.n_data]
        if cfg.task == 'pathfinding':
            self.target_paths = self.target_paths[:cfg.n_data]
        elif cfg.task == 'diameter':
            self.target_paths = self.target_diameters[:cfg.n_data]
        elif cfg.task == 'traveling':
            self.target_paths = self.target_travelings[:cfg.n_data]
        elif cfg.task == 'maze_gen':
            pass
            # tp = th.zeros_like(self.mazes_discrete)
            # tp[:, :, 4] = 1
            # tp[:, 4, :] = 1
            # self.target_paths = tp
        elif cfg.task == 'pathfinding_solnfree':
            # Inside the loss function, we'll compute path length.
            pass  

        else:
            raise Exception

        self.get_tiles(cfg.tileset.tiles)


def generate_random_maze(cfg):
    batch_size = 1
    width, height = cfg.width, cfg.height
    empty_chan, wall_chan, src_chan, trg_chan = Tiles.EMPTY, Tiles.WALL, Tiles.SRC, Tiles.TRG

    # Generate empty room with with borders.
    rand_maze_onehot = th.zeros((batch_size, Tiles.MAX + 1, width + 2, height + 2), dtype=int)
    rand_maze_onehot[:, wall_chan, [0, -1], :] = 1
    rand_maze_onehot[:, wall_chan, :, [0, -1]] = 1

    # Randomly generate wall/empty tiles.
    rand_walls = (th.rand((batch_size, 1, width, height)) < 0.5).int()
    rand_maze_onehot[:, wall_chan: wall_chan + 1, 1: -1, 1: -1] = rand_walls
    rand_maze_onehot[:, empty_chan: empty_chan + 1, 1: -1, 1: -1] = (rand_walls == 0).int()

    # Randomly generate sources/targets.
    src_xs, trg_xs = th.randint(0, width, (2, batch_size)) + 1
    src_ys, trg_ys = th.randint(0, height, (2, batch_size)) + 1
    while th.any((src_xs == trg_xs) & (src_ys == trg_ys)):
        src_xs, trg_xs = th.randint(0, width, (2, batch_size)) + 1
        src_ys, trg_ys = th.randint(0, height, (2, batch_size)) + 1

    # Remove wall/empty tiles at location of source/target.
    rand_maze_onehot[:, empty_chan, src_xs, src_ys] = 0
    rand_maze_onehot[:, wall_chan, src_xs, src_ys] = 0
    rand_maze_onehot[:, empty_chan, trg_xs, trg_ys] = 0
    rand_maze_onehot[:, wall_chan, trg_xs, trg_ys] = 0

    # Add sources and targets.
    rand_maze_onehot[:, src_chan, src_xs, src_ys] = 1
    rand_maze_onehot[:, trg_chan, trg_xs, trg_ys] = 1

    # Add TSP destinations.
    # n_dest = 4
    # dest_idxs = th.randint(0, width, (2, n_dest, batch_size))
    # rand_maze_onehot[:, Tiles.WALL, dest_idxs[0], dest_idxs[1]] = 1
    # rand_maze_onehot[:, Tiles.DEST, dest_idxs[0], dest_idxs[1]] = 1

    return rand_maze_onehot


def render_multihot(arr, cfg: Config, target_path: th.Tensor = None):
    """_summary_

    Args:
        arr (th.Tensor): A multi-hot array representing a map in the given domain. May have a variable number of 
            channels depending on what tile types are relevant.
        tiles (Iterable[int]): A list of tile types. Where the position of a tile type in the list corresponds to the 
            position of its channel in the multihot array.
        target_pa

    Returns:
        np.ndarray: An array corresponding to an RGB rendering of the map.
    """
    tileset: Tilesets.Tileset = cfg.tileset
    # TODO: Pull this out into a global variable or some such.
    empty_chan, wall_chan, src_chan, trg_chan = Tiles.EMPTY, Tiles.WALL, Tiles.SRC, Tiles.TRG
    empty_color = th.Tensor([1.0, 1.0, 1.0])
    wall_color = th.Tensor([0.0, 0.0, 0.0])
    src_color = th.Tensor([1.0, 1.0, 0.0])
    trg_color = th.Tensor([0.0, 1.0, 0.0])
    dest_color = th.Tensor([1.0, 1.0, 0.0])
    path_color = th.Tensor([1.0, 0.0, 0.0])
    colors = {empty_chan: empty_color, wall_chan: wall_color, Tiles.DEST: dest_color}
    colors.update({src_chan: src_color, trg_chan: trg_color, })
    colors.update({path_chan: path_color, })
    batch_size, n_chan, height, width = arr.shape
    im = th.zeros((batch_size, height, width, 3), dtype=th.float32)

    # img = PIL.Image.fromarray(np.uint8(im[0].cpu().numpy()))
    # Save image
    # img.save(cfg.log_dir + '/' + 'maze.png')

    if target_path is None:
        render_path = None
    else:
        render_path = target_path.clone()

    for tile in tileset.tiles:
        idxs = th.where(arr[:, tileset.tile_idxs[tile], ...] == 1)
        im[idxs[0], idxs[1], idxs[2], :] = colors[tile]

        # Path will not overwrite e.g. sources, destinations
        if tile != Tiles.EMPTY and render_path is not None:
            render_path[idxs[0], idxs[1], idxs[2]] = 0

    if render_path is not None:
        idxs = th.where(render_path == 1)
        # wtf?
        im[idxs[0], idxs[1], idxs[2], :] = src_color


    im = im.cpu().numpy()

    return im


def render_discrete(arr, cfg):
    # DEPRECATED
    empty_chan, wall_chan, src_chan, trg_chan = Tiles.EMPTY, Tiles.WALL, Tiles.SRC, Tiles.TRG
    empty_color = th.Tensor([1.0, 1.0, 1.0])
    wall_color = th.Tensor([0.0, 0.0, 0.0])
    src_color = th.Tensor([1.0, 1.0, 0.0])
    trg_color = th.Tensor([0.0, 1.0, 0.0])
    dest_color = th.Tensor([0.0, 1.0, 1.0])
    path_color = th.Tensor([1.0, 0.0, 0.0])
    colors = {empty_chan: empty_color, wall_chan: wall_color, Tiles.DEST: dest_color}
    colors.update({src_chan: src_color, trg_chan: trg_color, })
    colors.update({path_chan: path_color, })
    batch_size, height, width = arr.shape
    im = th.zeros((batch_size, height, width, 3), dtype=th.float32)

    # img = PIL.Image.fromarray(np.uint8(im[0].cpu().numpy()))
    # Save image
    # img.save(cfg.log_dir + '/' + 'maze.png')
    # TT()

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


def get_graph(onehot):
    # TODO: can all be onehot
    src, trg = None, None
    graph = nx.Graph()
    width, height = onehot.shape[1:]
    size = width * height
    graph.add_nodes_from(range(size))
    edges = []
    edge_feats = []
    # ret = scipy.sparse.csgraph.floyd_warshall(dist)
    for u in range(size):
        ux, uy = u // width, u % width
        if onehot[Tiles.WALL, ux, uy] == 1:
            continue
        # Checking if we care about src/trg based on number of channels (eek)
        if Tiles.SRC < onehot.shape[0] and onehot[Tiles.SRC, ux, uy] == 1:
            assert src is None
            src = u
        if Tiles.TRG < onehot.shape[0] and onehot[Tiles.TRG, ux, uy] == 1:
            assert trg is None
            trg = u
        neighbs_xy = [(ux - 1, uy), (ux, uy-1), (ux+1, uy), (ux, uy+1)]
        adj_feats = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        neighbs = [x * width + y for x, y in neighbs_xy]
        for v, (vx, vy), edge_feat in zip(neighbs, neighbs_xy, adj_feats):
            if not 0 <= v < size or onehot[Tiles.WALL, vx, vy] == 1:
                continue
            graph.add_edge(u, v)
            edges.append((u, v))
            edge_feats.append(edge_feat)
        edges.append((u, u))
        edge_feats.append((0, 0))
    # assert not (src is None and src == trg)
    # if src is None and src == trg:
        # TT()
    edges = th.Tensor(edges).long()
    edge_feats = th.Tensor(edge_feats).long()
    edges = rearrange(edges, 'e ij -> ij e')
    return graph, edges, edge_feats, src, trg


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


def bfs_nx(width, graph, src, trg):
    shortest_paths = dict(nx.shortest_path(graph, src))
    return None if trg not in shortest_paths else [(u // width, u % width) for u in shortest_paths[trg]]


def diameter(width, graph, traveling=True):
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

    # Place destinations in largest connected component.
    traveling_xy, dests_xy = None, None
    if traveling:
        dests = [max_connected.popitem()[0] for i in range(min(len(max_connected), 10))]
        target_traveling = nx.algorithms.approximation.traveling_salesman_problem(graph, nodes=dests, cycle=True)
        traveling_xy = [(u // width, u % width) for u in target_traveling]
        dests_xy = np.array([(u // width, u % width) for u in dests])
    return max_path_xy, max_connected_xy, traveling_xy, dests_xy


def get_shortest_path(onehot):
    width, height = onehot.shape[1:]
    graph, edges, edge_feats, src, trg = get_graph(onehot)
    assert (src is not None) and (trg is not None) and (trg != src)
    # srcs = th.argwhere(arr==Tiles.EMPTY)
    # src = srcs[np.random.randint(len(srcs))]
    # src = src[0] * width + src[1]
    # src = src.item()
    paths = dict(nx.shortest_path(graph, src))
    if trg in paths:
        path = paths[trg]
    else:
        # trg, path = paths.popitem()
        path = []
    path_xy = [(i // width, i % width) for i in path]
    return path_xy, edges, edge_feats


def get_target_path(maze_onehot, cfg):
    if cfg.task == 'pathfinding':
        # TODO: need to constrain mutation of sources and targets for this task.
        sol, edges, edge_feats = get_shortest_path(maze_onehot)
        # for (x, y) in sol:
            # offspring_target_paths[mi, x, y] = 1
        return sol, edges, edge_feats


def get_target_diam(maze_onehot, cfg):
    graph, edges, edge_feats, src, trg = get_graph(maze_onehot.cpu().numpy())
    diam, connected, traveling, dests_xy = diameter(maze_onehot.shape[-2], graph, traveling=cfg.task=='traveling')
    # for (x, y) in diam:
        # offspring_target_paths[mi, x, y] = 1
    return diam, traveling, edges, edge_feats


if __name__ == "__main__":
    main_mazes()
