#@title Imports and Notebook Utilities
#%tensorflow_version 2.x

import argparse
import os
from pdb import set_trace as TT
import sys

import numpy as np
import matplotlib.pyplot as pl
import PIL.Image, PIL.ImageDraw
import base64
import torch
from tqdm import tqdm_notebook, tnrange

from config import Config
from mazes import gen_rand_mazes, render_discrete
from model import CA
from training import train
from utils import VideoWriter, gen_pool, get_mse_loss, to_path


os.system('nvidia-smi -L')
if torch.cuda.is_available():
    print('Using GPU/CUDA.')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print('Not using GPU/CUDA, using CPU.')
    
np.set_printoptions(threshold=sys.maxsize)


def main():
  cfg = Config()
  args = argparse.ArgumentParser()
  args.add_argument('--load', action='store_true')
  args.add_argument('--render', action='store_true')
  args = args.parse_args()

  solvable_mazes_discrete, solvable_mazes_onehot, target_paths = gen_rand_mazes(data_n=cfg.data_n)
  rand_maze_ims = [render_discrete(rand_maze_discrete[None,])[0] for rand_maze_discrete in solvable_mazes_discrete]
  fig, ax = pl.subplots(figsize=(20, 5))
  pl.imshow(np.hstack(rand_maze_ims[:cfg.render_minibatch_size]))

  path_chan = 4
  assert path_chan == solvable_mazes_discrete.max() + 1
  solved_mazes = solvable_mazes_discrete.clone()
  solved_mazes[torch.where(target_paths == 1)] = path_chan
  solved_maze_ims = render_discrete(solved_mazes)
  fig, ax = pl.subplots(figsize=(20, 5))
  pl.imshow(np.hstack(solved_maze_ims[:cfg.render_minibatch_size]))

  fig, ax = pl.subplots(figsize=(20, 5))
  pl.imshow(np.hstack(target_paths[:cfg.render_minibatch_size].cpu()))
  pl.tight_layout()

  n_in_chan = 4  # The number of channels in the one-hot encodings of the training mazes.
  assert n_in_chan == solvable_mazes_onehot.shape[1]

  # setup training
  ca = CA(n_in_chan, cfg.n_aux_chan, cfg.n_hidden_chan) 
  param_n = sum(p.numel() for p in CA().parameters())
  print('CA param count:', param_n)
  opt = torch.optim.Adam(ca.parameters(), 1e-4)


  if args.load:
    ca.load_state_dict(torch.load(f'{cfg.log_dir}/ca_state_dict.pt'))
    opt.load_state_dict(torch.load(f'{cfg.log_dir}/opt_state_dict.pt'))
    print(f'Loaded CA state dict and optimizer state dict from {cfg.log_dir}.')

  # with torch.no_grad():
  #   pool = ca.seed(256)

  #@title training loop {vertical-output: true}

  # The set of initial mazes (padded with 0s, channel-wise, to act as the initial state of the CA).
  training_mazes_onehot, training_mazes_discrete = solvable_mazes_onehot, solvable_mazes_discrete
  # training_maze_xs = torch.zeros(training_mazes.shape[0], n_aux_chan, training_mazes.shape[2], training_mazes.shape[3])
  # training_maze_xs[:, :n_in_chan, :, :] = training_mazes
  # pl.imshow(np.hstack(training_maze_xs[...,2:3,:,:].permute([0,2,3,1]).cpu()))
  # sys.exit()
  # Images of solved mazes, for reference.
  solved_maze_ims = torch.Tensor(solved_maze_ims)

  if args.render:
    render_trained(ca, training_mazes_onehot, training_mazes_discrete, cfg)
  else:
    train(ca, opt, training_mazes_onehot, solved_maze_ims, target_paths, cfg)


def render_trained(ca, mazes_onehot, mazes_discrete, cfg):
  #@title NCA video {vertical-output: true}

  pool = gen_pool(mazes_onehot.shape[0], ca.n_out_chan, mazes_onehot.shape[2], mazes_onehot.shape[3])
  batch_idx = np.random.choice(pool.shape[0], cfg.render_minibatch_size, replace=False)
  render_batch_idx = batch_idx[:cfg.render_minibatch_size]
  x = pool[batch_idx]
  x0 = mazes_onehot[batch_idx]
  ca.reset(x0)

  with VideoWriter() as vid, torch.no_grad():

    def write(x, x0):
      img = to_path(x[:cfg.render_minibatch_size])[...,None].cpu()
      img = (img - img.min()) / (img.max() - img.min())
      img = np.hstack(img)
      # solved_maze_ims = [render_discrete(x0i.argmax(0)[None,...]) for x0i in x0]
      # solved_mazes_im = np.hstack(np.vstack(solved_maze_ims))
      solved_mazes_im = np.hstack(render_discrete(mazes_discrete[render_batch_idx]))
      imgs = np.vstack([solved_mazes_im, np.tile(img, (1, 1, 3))])
      vid.add(imgs)
  #   fig, ax = pl.subplots(figsize=(10,10))
  #   pl.imshow(imgs)
  #   pl.show()

    write(x, x0)
    # for k in tnrange(300, leave=False):
    # for k in range(m):
      # step_n = min(2**(k//30), 16)
    for i in range(cfg.expected_net_steps):
      x = ca(x)
      # pl.imshow(x)
      # pl.show()
      write(x, x0)
  #     if i < expected_net_steps - 1:
  #       clear_output(True)
      
      # Test trained model on newly-generated solvable mazes to test inference.

def evaluate(ca, cfg):
  n_test_minibatches = 10
  n_test_mazes = n_test_minibatches * cfg.minibatch_size
  test_mazes_discrete, test_mazes_onehot, target_paths = gen_rand_mazes(data_n=n_test_mazes)

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
                  target_path_ims = np.tile(np.hstack(target_paths_mini_batch[:cfg.render_minibatch_size].cpu())[...,None], (1, 1, 3))
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