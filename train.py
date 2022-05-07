from pdb import set_trace as TT
import pickle
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer
import torch as th
import torchvision.models as models
from tqdm import tqdm
from config import ClArgsConfig
from evaluate import evaluate
from mazes import Mazes, render_discrete
from models.gnn import GCN
from models.nn import PathfindingNN

from utils import get_discrete_loss, get_mse_loss, Logger, to_path, save


def train(model: PathfindingNN, opt: th.optim.Optimizer, maze_data: Mazes, maze_data_val: Mazes, 
        target_paths: th.Tensor, logger: Logger, cfg: ClArgsConfig):
    mazes_onehot, maze_ims = maze_data.mazes_onehot, maze_data.maze_ims
    minibatch_size = min(cfg.minibatch_size, cfg.n_data)
    lr_sched = th.optim.lr_scheduler.MultiStepLR(opt, [10000], 0.1)
    logger.last_time = timer()
    hid_states = model.seed(batch_size=mazes_onehot.shape[0])

    for i in tqdm(range(logger.n_step, cfg.n_updates)):
        with th.no_grad():
            # Randomly select indices of data-points on which to train during this update step (i.e., minibatch)
            batch_idx = np.random.choice(hid_states.shape[0], minibatch_size, replace=False)
            render_batch_idx = batch_idx[:cfg.render_minibatch_size]

            # x = th.zeros(x_maze.shape[0], n_chan, x_maze.shape[2], x_maze.shape[3])
            # x[:, :4, :, :] = x_maze
            x0 = mazes_onehot[batch_idx].clone()
            target_paths_mini_batch = target_paths[batch_idx]

            model.reset(x0)

        x = hid_states[batch_idx]

        # step_n = np.random.randint(32, 96)

        # TODO: move initial auxiliary state to model? Probably a better way...
        # Ad hoc:
        # if cfg.model == "MLP":
            # assert not cfg.shared_weights    # TODO
            # x = x0    # tehe

        # else:
        # FYI: this is from the differentiating NCA textures notebook. Weird checkpointing behavior indeed! See the
        #   comments about the `sparse_updates` arg. -SE
        # The following line is equivalent to this code:
        # for k in range(cfg.n_layers):
            # x = model(x)
        # It uses gradient checkpointing to save memory, which enables larger
        # batches and longer CA step sequences. Surprisingly, this version
        # is also ~2x faster than a simple loop, even though it performs
        # the forward pass twice!
        # x = th.utils.checkpoint.checkpoint_sequential([model]*cfg.n_layers, 32, x)

        loss = 0

        for _ in range(cfg.n_layers // cfg.loss_interval):

            # Hackish way of storing gradient only at last "chunk" (with checkpointing), or not.
            if cfg.sparse_update:
                x = th.utils.checkpoint.checkpoint_sequential([model]*cfg.loss_interval, min(16, cfg.loss_interval), x)
            else:
                for _ in range(cfg.n_layers):
                    x = model(x)

            loss += get_mse_loss(x, target_paths_mini_batch)

        discrete_loss = get_discrete_loss(x, target_paths_mini_batch).float().mean()

        with th.no_grad():
            if "Fixed" not in cfg.model:

                # NOTE: why are we doing this??
                model.reset(x0)

                loss.backward()

                # DEBUG: gradient only collected for certain "chunks" of the network as a result of gradient checkpointing.
                # for name, p in model.named_parameters():
                    # print(name, "Grad is None?", p.grad is None)

                # for p in model.parameters():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        assert cfg.model == "BfsNCA"
                        continue
                    if not isinstance(model, GCN) and cfg.cut_conv_corners and "weight" in name:
                        # Zero out all the corners
                        p.grad[:, :, 0, 0] = p.grad[:, :, -1, -1] = p.grad[:, :, -1, 0] = p.grad[:, :, 0, -1] = 0
                    p.grad /= (p.grad.norm()+1e-8)     # normalize gradients 
                opt.step()
                opt.zero_grad()
            # lr_sched.step()
            logger.log(loss=loss.item(), discrete_loss=discrete_loss.item())
                    
            if i % cfg.save_interval == 0 or i == cfg.n_updates - 1:
                save(model, opt, maze_data, logger, cfg)
                # print(f'Saved CA and optimizer state dicts and maze archive to {cfg.log_dir}')

            if i % cfg.eval_interval == 0:
                val_stats = evaluate(model, maze_data_val, cfg.val_batch_size, "validate", cfg)
                logger.log_val(val_stats)

            if i % cfg.log_interval == 0 or i == cfg.n_updates - 1:
                log(logger, lr_sched, maze_ims, x, target_paths, render_batch_idx, cfg)


def log(logger, lr_sched, maze_ims, x, target_paths, render_batch_idx, cfg):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplot(411)
        # smooth_loss_log = smooth(logger.loss_log, 10)
    loss_log = np.array(logger.loss_log)
    discrete_loss_log = logger.discrete_loss_log
    discrete_loss_log = np.where(np.array(discrete_loss_log) == 0, 1e-8, discrete_loss_log)
    plt.plot(loss_log, '.', alpha=0.1, label='loss')
    plt.plot(discrete_loss_log, '.', alpha=0.1, label='discrete loss')
    val_loss = logger.get_val_stat('losses')
    val_discrete_loss = logger.get_val_stat('disc_losses')
    plt.plot(list(val_loss.keys()), [v[0] for v in list(val_loss.values())], '.', alpha=1.0, label='val loss')
    plt.plot(list(val_discrete_loss.keys()), [v[0] for v in list(val_discrete_loss.values())], '.', alpha=1.0, label='val discrete loss')
    plt.legend()
    plt.yscale('log')
    # plt.ylim(np.min(np.hstack((loss_log, discrete_loss_log))), logger.loss_log[0])
    plt.ylim(np.min(np.hstack((loss_log, discrete_loss_log))), 
             np.max(np.hstack((loss_log, discrete_loss_log, [v[0] for v in list(val_loss.values())]))))
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
    print('step_n:', len(logger.loss_log),
        ' loss: {:.6e}'.format(logger.loss_log[-1]),
        ' fps: {:.2f}'.format(fps), 
        ' lr:', lr_sched.get_last_lr(), # end=''
        )
    logger.last_time = timer()


def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
