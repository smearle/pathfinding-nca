from logging import Logger
from pdb import set_trace as TT
import pickle
from matplotlib import pyplot as pl
import numpy as np
from timeit import default_timer as timer
import torch
import torchvision.models as models
from mazes import Mazes

from utils import gen_pool, get_mse_loss, to_path, save


def train(ca, opt, maze_data, target_paths, logger, cfg):
  mazes_onehot, maze_ims = maze_data.mazes_onehot, maze_data.maze_ims
  # Upper bound of net steps = step_n * m. Expected net steps = (minibatch_size / data_n) * m * step_n. (Since we select from pool
  # randomly before each mini-episode.)
  minibatch_size = min(cfg.minibatch_size, cfg.data_n)
  m = int(cfg.expected_net_steps / cfg.step_n * cfg.data_n / minibatch_size)
  # m = expected_net_steps
  lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [10000], 0.1)
  last_time = timer()

  for i in range(cfg.n_updates):
    with torch.no_grad():

      if cfg.gen_new_data_interval and i % cfg.gen_new_data_interval == 0:
        maze_data = Mazes(cfg)
        mazes_onehot, maze_ims = maze_data.mazes_onehot, maze_data.maze_ims

      if i % m == 0:
        # pool = training_maze_xs.clone()
        pool = gen_pool(mazes_onehot.shape[0], ca.n_out_chan, mazes_onehot.shape[2], mazes_onehot.shape[3])

      # batch_idx = np.random.choice(len(pool), 4, replace=False)

      # Randomly select indices of data-points on which to train during this update step (i.e., minibatch)
      batch_idx = np.random.choice(pool.shape[0], minibatch_size, replace=False)
      render_batch_idx = batch_idx[:cfg.render_minibatch_size]

      x = pool[batch_idx]
      # x = torch.zeros(x_maze.shape[0], n_chan, x_maze.shape[2], x_maze.shape[3])
      # x[:, :4, :, :] = x_maze
      x0 = mazes_onehot[batch_idx].clone()
      target_paths_mini_batch = target_paths[batch_idx]

      ca.reset(x0)

    # step_n = np.random.randint(32, 96)

    # The following line is equivalent to this code:
    # for k in range(step_n):
      # x = ca(x)
    # It uses gradient checkpointing to save memory, which enables larger
    # batches and longer CA step sequences. Surprisingly, this version
    # is also ~2x faster than a simple loop, even though it performs
    # the forward pass twice!
    x = torch.utils.checkpoint.checkpoint_sequential([ca]*cfg.step_n, 16, x)
    
    loss = get_mse_loss(x, target_paths_mini_batch)

    with torch.no_grad():
      loss.backward()
      for p in ca.parameters():
        p.grad /= (p.grad.norm()+1e-8)   # normalize gradients 
      opt.step()
      opt.zero_grad()
      # lr_sched.step()
      pool[batch_idx] = x                # update pool
      
      logger.log(loss=loss.item())

      if i % cfg.log_interval == 0:
        fig, ax = pl.subplots(2, 4, figsize=(20, 10))
        pl.subplot(411)
        # smooth_loss_log = smooth(logger.loss_log, 10)
        smooth_loss_log = logger.loss_log
        pl.plot(smooth_loss_log, '.', alpha=0.1)
        pl.yscale('log')
        pl.ylim(np.min(smooth_loss_log), logger.loss_log[0])
        # imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
        render_paths = to_path(x[:cfg.render_minibatch_size]).cpu()
        # imshow(np.hstack(imgs))
        pl.subplot(412)
        pl.imshow(np.hstack(maze_ims[render_batch_idx].cpu()))
        pl.subplot(413)
        pl.imshow(np.hstack(render_paths))  #, vmin=-1.0, vmax=1.0)
        pl.subplot(414)
        pl.imshow(np.hstack(target_paths[render_batch_idx].cpu()))
        # pl.imshow(np.hstack(x[...,-2:-1,:,:].permute([0,2,3,1]).cpu()))
        # pl.imshow(np.hstack(ca.x0[...,0:1,:,:].permute([0,2,3,1]).cpu()))
        # print(f'path activation min: {render_paths.min()}, max: {render_paths.max()}')
        pl.savefig(f'{cfg.log_dir}/training_progress.png')
        pl.close()
          
      if i % cfg.save_interval == 0:
        save(ca, opt, maze_data, logger, cfg)

        # print(f'Saved CA and optimizer state dicts and maze archive to {cfg.log_dir}')

      if i % cfg.log_interval == 0:
        fps = cfg.step_n * cfg.minibatch_size / (timer() - last_time) if i > 0 else 0
        print('\rstep_n:', len(logger.loss_log),
          ' loss: {:.6e}'.format(loss.item()),
          ' fps: {:.2f}'.format(fps), 
          ' lr:', lr_sched.get_last_lr(), end=''
          )
        last_time = timer()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth