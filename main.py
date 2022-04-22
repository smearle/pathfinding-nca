#@title Imports and Notebook Utilities
#%tensorflow_version 2.x

import argparse
import json
import os
from pdb import set_trace as TT
import pickle
import shutil
import sys
from matplotlib import animation

import numpy as np
import matplotlib.pyplot as pl
import PIL.Image, PIL.ImageDraw
import base64
import torch
import torchinfo
from tqdm import tqdm_notebook, tnrange

from config import ClArgsConfig
from mazes import gen_rand_mazes, load_dataset, render_discrete, Mazes
from models import NCA, GCN, MLP
from training import train
from utils import Logger, VideoWriter, gen_pool, get_mse_loss, to_path, load


np.set_printoptions(threshold=sys.maxsize)


def main():
  os.system('nvidia-smi -L')
  if torch.cuda.is_available():
      print('Using GPU/CUDA.')
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
      print('Not using GPU/CUDA, using CPU.')
    
  cfg = ClArgsConfig()
  cfg.n_in_chan = 4  # The number of channels in the one-hot encodings of the training mazes.
  model_cls = globals()[cfg.model]
  
  # Setup training
  model = model_cls(cfg)
  # Set a dummy initial maze state.
  model.reset(torch.zeros(cfg.minibatch_size, cfg.n_in_chan, cfg.width, cfg.width))

  # FIXME: ad hoc (?)
  if cfg.model == "MLP":
    torchinfo.summary(model, input_size=(cfg.minibatch_size, cfg.n_in_chan, (cfg.width + 2), (cfg.width + 2)))

  else:
    torchinfo.summary(model, input_size=(cfg.minibatch_size, cfg.n_hid_chan, cfg.width, (cfg.width + 2)))

  # param_n = sum(p.numel() for p in model.parameters())
  # print('model param count:', param_n)
  opt = torch.optim.Adam(model.parameters(), cfg.learning_rate)
  try:
    maze_data_train, maze_data_val, _ = load_dataset(cfg.n_data, cfg.device)
  except FileNotFoundError as e:
    print("No maze data files found. Run `python mazes.py` to generate the dataset.")
    raise
  maze_data_val.get_subset(cfg.n_val_data)
  # cfg.n_data = maze_data_train.mazes_onehot.shape[0]

  if cfg.load:
    model, opt, logger = load(model, opt, cfg)

  else:
    if cfg.overwrite and os.path.exists(cfg.log_dir):
      shutil.rmtree(cfg.log_dir)
    try:
      os.mkdir(cfg.log_dir)
    except FileExistsError:
      print(f"Experiment log folder {cfg.log_dir} already exists. Use `--load` or `--overwrite` command line arguments "\
      "to load or overwrite it.")
    logger = Logger()

  # Save a dictionary of the config to a json for future reference.
  json.dump(cfg.__dict__, open(f'{cfg.log_dir}/config.json', 'w'))

  mazes_onehot, mazes_discrete, maze_ims, target_paths = \
    (maze_data_train.mazes_onehot, maze_data_train.mazes_discrete, maze_data_train.maze_ims, maze_data_train.target_paths)

# fig, ax = pl.subplots(figsize=(20, 5))
# pl.imshow(np.hstack(maze_ims[:cfg.render_minibatch_size]))

  path_chan = 4
  assert path_chan == mazes_discrete.max() + 1
# solved_mazes = mazes_discrete.clone()
# solved_mazes[torch.where(target_paths == 1)] = path_chan
# fig, ax = pl.subplots(figsize=(20, 5))
# pl.imshow(np.hstack(solved_maze_ims[:cfg.render_minibatch_size]))

# fig, ax = pl.subplots(figsize=(20, 5))
# pl.imshow(np.hstack(target_paths[:cfg.render_minibatch_size].cpu()))
# pl.tight_layout()

  assert cfg.n_in_chan == mazes_onehot.shape[1]

  # The set of initial mazes (padded with 0s, channel-wise, to act as the initial state of the CA).

  if cfg.render:
    # TT()
    render_trained(model, maze_data_train, cfg)
  else:
    train(model, opt, maze_data_train, maze_data_val, target_paths, logger, cfg)


def render_trained(ca, maze_data, cfg, pyplot_animation=True):
  """Generate a video showing the behavior of the trained NCA on mazes from its training distribution.
  """
  mazes_onehot, mazes_discrete = maze_data.mazes_onehot, maze_data.mazes_discrete
  pool = gen_pool(mazes_onehot.shape[0], ca.n_out_chan, mazes_onehot.shape[2], mazes_onehot.shape[3])
  render_minibatch_size = min(cfg.render_minibatch_size, mazes_onehot.shape[0])
  batch_idx = np.random.choice(pool.shape[0], render_minibatch_size, replace=False)
  x = pool[batch_idx]
  x0 = mazes_onehot[batch_idx]
  ca.reset(x0)

  if pyplot_animation:
    fig, ax = pl.subplots(figsize=(10,10))


    def get_imgs(x, x0):
      img = to_path(x[:cfg.render_minibatch_size])[...,None].cpu()
      img = (img - img.min()) / (img.max() - img.min())
      img = np.hstack(img.detach())
      # solved_maze_ims = [render_discrete(x0i.argmax(0)[None,...]) for x0i in x0]
      # solved_mazes_im = np.hstack(np.vstack(solved_maze_ims))
      solved_mazes_im = np.hstack(render_discrete(mazes_discrete[batch_idx]))
      imgs = np.vstack([solved_mazes_im, np.tile(img, (1, 1, 3))])

      return imgs


    imgs = get_imgs(x, x0)
    pl_im = pl.imshow(imgs, interpolation='none')

    def init():
      pl_im.set_data(imgs)

      return pl_im,

    xs = []

    with torch.no_grad():
      xs.append(x)
      for i in range(cfg.expected_net_steps):
        x = ca(x)
        xs.append(x)


    def animate(i):
      xi = xs[i]
      imgs = get_imgs(xi, x0)
      pl_im.set_data(imgs)
      # line.set_ydata(np.sin(x + i / 50))  # update the data.
      # pl.show()

      return pl_im,


    anim = animation.FuncAnimation(
      fig, animate, init_func=init, interval=1, frames=cfg.expected_net_steps, blit=True, save_count=50)
    anim.save(f'{cfg.log_dir}/path_nca_anim.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    # pl.show()

  else:

    with VideoWriter(filename=f'{cfg.log_dir}/path_nca.mp4') as vid, torch.no_grad():
      vid.add(imgs)

      with torch.no_grad():

        for i in range(cfg.expected_net_steps):
          x = ca(x)
          imgs = get_imgs(x, x0)
          vid.add(imgs)
          

def evaluate_train(ca, cfg):
  """Evaluate the trained model on the training set."""
  # TODO: 
  pass

def evaluate_test(ca, cfg):
  """Evaluate the trained model on a test set."""
  n_test_minibatches = 10
  n_test_mazes = n_test_minibatches * cfg.minibatch_size
  _, _, maze_data_test = load_dataset(cfg.n_eval_data, cfg.device)
  test_mazes_onehot, test_mazes_discrete, target_paths = maze_data_test.mazes_onehot, maze_data_test.mazes_discrete, \
    maze_data_test.target_paths

  with torch.no_grad():
      test_losses = []
      i = 0

      for i in range(n_test_minibatches):
          batch_idx = np.arange(i*cfg.minibatch_size, (i+1)*cfg.minibatch_size, dtype=int)
          x0 = test_mazes_onehot[batch_idx]
          x0_discrete = test_mazes_discrete[batch_idx]
          x = gen_pool(size=cfg.minibatch_size, n_chan=ca.n_out_chan, height=x0_discrete.shape[2], width=x0_discrete.shape[3])
          target_paths_mini_batch = target_paths[batch_idx]
          ca.reset(x0)

          for j in range(cfg.expected_net_steps):
              x = ca(x)

              if j == cfg.expected_net_steps - 1:
                  test_loss = get_mse_loss(x, target_paths_mini_batch).item()
                  test_losses.append(test_loss)
                  # clear_output(True)
                  fig, ax = pl.subplots(figsize=(10, 10))
                  solved_maze_ims = np.hstack(render_discrete(x0_discrete[:cfg.render_minibatch_size]))
                  target_path_ims = np.tile(np.hstack(
                    target_paths_mini_batch[:cfg.render_minibatch_size].cpu())[...,None], (1, 1, 3)
                    )
                  predicted_path_ims = to_path(x[:cfg.render_minibatch_size])[...,None].cpu()
                  # img = (img - img.min()) / (img.max() - img.min())
                  predicted_path_ims = np.hstack(predicted_path_ims)
                  predicted_path_ims = np.tile(predicted_path_ims, (1, 1, 3))
                  imgs = np.vstack([solved_maze_ims, target_path_ims, predicted_path_ims])
                  pl.imshow(imgs)
                  pl.show()

  print(f'Mean test loss: {np.mean(test_losses)}') 


if __name__ == '__main__':
  main()