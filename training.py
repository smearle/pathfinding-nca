from logging import Logger
from pdb import set_trace as TT
import pickle
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer
import torch
import torchvision.models as models
from config import ClArgsConfig
from mazes import Mazes, render_discrete
from models.nn import PathfindingNN

from utils import gen_pool, get_mse_loss, to_path, save


def train(model: PathfindingNN, opt, maze_data, maze_data_val, target_paths, logger, cfg: ClArgsConfig):
    mazes_onehot, maze_ims = maze_data.mazes_onehot, maze_data.maze_ims
    # Upper bound of net steps = step_n * m. Expected net steps = (minibatch_size / data_n) * m * step_n. (Since we select from pool
    # randomly before each mini-episode.)
    minibatch_size = min(cfg.minibatch_size, cfg.n_data)
    m = int(cfg.n_layers / cfg.n_layers * cfg.n_data / minibatch_size)
    # m = n_layers
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [10000], 0.1)
    logger.last_time = timer()
    hid_states = gen_pool(mazes_onehot.shape[0], cfg.n_hid_chan, mazes_onehot.shape[2], mazes_onehot.shape[3])

    for i in range(logger.n_step, cfg.n_updates):
        with torch.no_grad():
            # Randomly select indices of data-points on which to train during this update step (i.e., minibatch)
            batch_idx = np.random.choice(hid_states.shape[0], minibatch_size, replace=False)
            render_batch_idx = batch_idx[:cfg.render_minibatch_size]

            x = hid_states[batch_idx]
            # x = torch.zeros(x_maze.shape[0], n_chan, x_maze.shape[2], x_maze.shape[3])
            # x[:, :4, :, :] = x_maze
            x0 = mazes_onehot[batch_idx].clone()
            target_paths_mini_batch = target_paths[batch_idx]

            model.reset(x0)

        # step_n = np.random.randint(32, 96)

        # Ad hoc:
        if cfg.model == "MLP":
            assert not cfg.shared_weights    # TODO
            x = x0    # tehe

        if not cfg.shared_weights:
            x = model(x)

        else:
            # The following line is equivalent to this code:
            # for k in range(step_n):
                # x = model(x)
            # It uses gradient checkpointing to save memory, which enables larger
            # batches and longer CA step sequences. Surprisingly, this version
            # is also ~2x faster than a simple loop, even though it performs
            # the forward pass twice!
            x = torch.utils.checkpoint.checkpoint_sequential([model]*cfg.n_layers, 16, x)
        
        loss = get_mse_loss(x, target_paths_mini_batch)

        with torch.no_grad():
            if "Fixed" not in cfg.model:
                loss.backward()
                for p in model.parameters():
                    # TODO: ignore "corner" convolutional weights here if specified in config.
                    p.grad /= (p.grad.norm()+1e-8)     # normalize gradients 
                opt.step()
                opt.zero_grad()
            # lr_sched.step()
            logger.log(loss=loss.item())
                    
            if i % cfg.save_interval == 0:
                save(model, opt, maze_data, logger, cfg)

                # print(f'Saved CA and optimizer state dicts and maze archive to {cfg.log_dir}')

            if i % cfg.eval_interval == 0:
                val_loss = evaluate(model, maze_data_val, cfg.val_batch_size, cfg)
                logger.log_val(val_loss=val_loss)

            if i % cfg.log_interval == 0:
                log(logger, lr_sched, maze_ims, x, target_paths, render_batch_idx, cfg)


def log(logger, lr_sched, maze_ims, x, target_paths, render_batch_idx, cfg):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplot(411)
        # smooth_loss_log = smooth(logger.loss_log, 10)
    smooth_loss_log = logger.loss_log
    plt.plot(smooth_loss_log, '.', alpha=0.1)
    plt.plot(list(logger.val_loss_log.keys()), list(logger.val_loss_log.values()), '.', alpha=1.0)
    plt.yscale('log')
    plt.ylim(np.min(smooth_loss_log), logger.loss_log[0])
        # imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
    render_paths = to_path(x[:cfg.render_minibatch_size]).cpu()
        # imshow(np.hstack(imgs))
    plt.subplot(412)
    plt.imshow(np.hstack(maze_ims[render_batch_idx].cpu()))
    plt.subplot(413)
    plt.imshow(np.hstack(render_paths))    #, vmin=-1.0, vmax=1.0)
    plt.subplot(414)
    plt.imshow(np.hstack(target_paths[render_batch_idx].cpu()))
        # plt.imshow(np.hstack(x[...,-2:-1,:,:].permute([0,2,3,1]).cpu()))
        # plt.imshow(np.hstack(ca.x0[...,0:1,:,:].permute([0,2,3,1]).cpu()))
        # print(f'path activation min: {render_paths.min()}, max: {render_paths.max()}')
    plt.savefig(f'{cfg.log_dir}/training_progress.png')
    plt.close()
    fps = cfg.n_layers * cfg.minibatch_size * cfg.log_interval / (timer() - logger.last_time)
    print('\rstep_n:', len(logger.loss_log),
        ' loss: {:.6e}'.format(logger.loss_log[-1]),
        ' fps: {:.2f}'.format(fps), 
        ' lr:', lr_sched.get_last_lr(), end=''
        )
    logger.last_time = timer()


def evaluate(model, data, batch_size, cfg, render=False):
    # TODO: re-use this function when evaluating on training/test set, after training.
    """Evaluate the trained model on a dataset without collecting gradients."""
    mazes_onehot, mazes_discrete, target_paths = data.mazes_onehot, data.mazes_discrete, \
        data.target_paths

    with torch.no_grad():
        test_losses = []
        i = 0

        for i in range(mazes_onehot.shape[0] // batch_size):
            batch_idx = np.arange(i*batch_size, (i+1)*batch_size, dtype=int)
            x0 = mazes_onehot[batch_idx]
            x0_discrete = mazes_discrete[batch_idx]

            if cfg.model == "MLP":
                x = x0

            else:
                x = gen_pool(size=batch_size, n_chan=cfg.n_hid_chan, height=x0_discrete.shape[1], width=x0_discrete.shape[2])

            target_paths_mini_batch = target_paths[batch_idx]
            model.reset(x0)

            for j in range(cfg.n_layers):
                x = model(x)

                if j == cfg.n_layers - 1:
                    test_loss = get_mse_loss(x, target_paths_mini_batch).item()
                    test_losses.append(test_loss)
                    # clear_output(True)
                    if render:
                        fig, ax = plt.subplots(figsize=(10, 10))
                        solved_maze_ims = np.hstack(render_discrete(x0_discrete[:cfg.render_minibatch_size]))
                        target_path_ims = np.tile(np.hstack(
                            target_paths_mini_batch[:cfg.render_minibatch_size].cpu())[...,None], (1, 1, 3)
                            )
                        predicted_path_ims = to_path(x[:cfg.render_minibatch_size])[...,None].cpu()
                        # img = (img - img.min()) / (img.max() - img.min())
                        predicted_path_ims = np.hstack(predicted_path_ims)
                        predicted_path_ims = np.tile(predicted_path_ims, (1, 1, 3))
                        imgs = np.vstack([solved_maze_ims, target_path_ims, predicted_path_ims])
                        plt.imshow(imgs)
                        plt.show()

    mean_eval_loss = np.mean(test_losses)
    # print(f'Mean evaluation loss: {mean_eval_loss}') 

    return mean_eval_loss



def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
