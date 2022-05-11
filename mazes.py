import os
from pdb import set_trace as TT
import pickle
import numpy as np
from pathlib import Path
import torch

from config import ClArgsConfig


path_chan = 4
maze_data_fname = os.path.join(Path(__file__).parent, os.path.join("data", "maze_data"))
train_fname = f"{maze_data_fname}_train.pk"
val_fname = f"{maze_data_fname}_val.pk"
test_fname = f"{maze_data_fname}_test.pk"


def main_mazes(cfg):
    """Generate random mazes for training/testing."""
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


def load_dataset(n_data=None, device='cpu'):
    """Load the dataset of random mazes"""
    with open(train_fname, 'rb') as f:
        maze_data_train = pickle.load(f)
    maze_data_train.get_subset(n_data)

    with open(val_fname, 'rb') as f:
        maze_data_val = pickle.load(f)

    with open(test_fname, 'rb') as f:
        maze_data_test = pickle.load(f)
    maze_data_test.get_subset(n_data)

    return maze_data_train.to(device), maze_data_val.to(device), maze_data_test.to(device)


class Mazes():
    def __init__(self, cfg):
        width, height = cfg.width, cfg.height
        self.mazes_discrete, self.mazes_onehot, self.target_paths = gen_rand_mazes(cfg)
        self.maze_ims = torch.Tensor(np.array(
            [render_discrete(maze_discrete[None,], cfg)[0] for maze_discrete in self.mazes_discrete]
        ))
    
    def to(self, device):
        self.mazes_discrete, self.mazes_onehot, self.target_paths = self.mazes_discrete.to(device), \
            self.mazes_onehot.to(device), self.target_paths.to(device)
        return self

    def get_subset(self, n_data=None):
        if n_data is None:
            n_data = len(self.mazes_discrete)
        self.mazes_discrete, self.mazes_onehot, self.target_paths = self.mazes_discrete[:n_data], \
            self.mazes_onehot[:n_data], self.target_paths[:n_data]


def generate_random_maze(cfg):
    batch_size = 1
    width = 16
    empty_chan, wall_chan, src_chan, trg_chan = cfg.empty_chan, cfg.wall_chan, cfg.src_chan, cfg.trg_chan

    # Generate empty room with with borders.
    rand_maze_onehot = torch.zeros((batch_size, 4, width + 2, width + 2), dtype=int)
    rand_maze_onehot[:, wall_chan, [0, -1], :] = 1
    rand_maze_onehot[:, wall_chan, :, [0, -1]] = 1

    # Randomly generate wall/empty tiles.
    rand_walls = torch.randint(0, 2, (batch_size, 1, width, width))
    rand_maze_onehot[:, wall_chan: wall_chan + 1, 1: -1, 1: -1] = rand_walls
    rand_maze_onehot[:, empty_chan: empty_chan + 1, 1: -1, 1: -1] = (rand_walls == 0).int()

    # Randomly generate sources/targets.
    src_xs, src_ys, trg_xs, trg_ys = torch.randint(0, width, (4, batch_size)) + 1

    # Remove wall/empty tiles at location of source/target.
    rand_maze_onehot[torch.arange(batch_size), empty_chan, src_xs, src_ys] = 0
    rand_maze_onehot[torch.arange(batch_size), wall_chan, src_xs, src_ys] = 0
    rand_maze_onehot[torch.arange(batch_size), empty_chan, trg_xs, trg_ys] = 0
    rand_maze_onehot[torch.arange(batch_size), wall_chan, trg_xs, trg_ys] = 0

    # Add sources and targets.
    rand_maze_onehot[torch.arange(batch_size), src_chan, src_xs, src_ys] = 1
    rand_maze_onehot[torch.arange(batch_size), trg_chan, trg_xs, trg_ys] = 1

    return rand_maze_onehot


def render_discrete(arr, cfg):
    empty_chan, wall_chan, src_chan, trg_chan = cfg.empty_chan, cfg.wall_chan, cfg.src_chan, cfg.trg_chan
    empty_color = torch.Tensor([1.0, 1.0, 1.0])
    wall_color = torch.Tensor([0.0, 0.0, 0.0])
    src_color = torch.Tensor([1.0, 1.0, 0.0])
    trg_color = torch.Tensor([0.0, 1.0, 0.0])
    path_color = torch.Tensor([1.0, 0.0, 0.0])
    colors = {empty_chan: empty_color, wall_chan: wall_color, path_chan: path_color, src_chan: src_color, 
                trg_chan: trg_color, }
    batch_size, height, width = arr.shape
    im = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)

    for chan, color in colors.items():
        idxs = torch.where(arr == chan)
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


def bfs(arr, passable=0, impassable=1, src=2, trg=3):
    width = arr.shape[0]
    assert width == arr.shape[1]
    srcs = np.argwhere(arr == src)
    assert srcs.shape[0] == 1
    src = tuple(srcs[0])
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


def gen_rand_mazes(cfg):
    # Generate new random mazes until we get enough solvable ones.
    n_data = cfg.n_data
    solvable_mazes_onehot = []
    solvable_mazes_discrete = []
    target_paths = []
    i = 0
    while len(solvable_mazes_onehot) < n_data:
        rand_maze_onehot = generate_random_maze(cfg)
        rand_maze_discrete = rand_maze_onehot.argmax(axis=1)
        sol = bfs(rand_maze_discrete[0].cpu().numpy())
        if sol:
            solvable_mazes_onehot.append(rand_maze_onehot)
            solvable_mazes_discrete.append(rand_maze_discrete)
            target_path = torch.zeros_like(rand_maze_discrete)
            for x, y in sol:
                    target_path[0, x, y] = 1
            target_paths.append(target_path)
        i += 1
    # print(f'Solution length: {len(sol)}')
    print(f'Generated {i} random mazes to produce {n_data} solvable mazes.')

    return torch.vstack(solvable_mazes_discrete), torch.vstack(solvable_mazes_onehot), torch.vstack(target_paths)


if __name__ == "__main__":
    cfg = ClArgsConfig()
    main_mazes(cfg)
