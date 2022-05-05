#@title Imports and Notebook Utilities
#%tensorflow_version 2.x

import json
import os
from pdb import set_trace as TT
import shutil
import sys
from matplotlib import animation

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchinfo

from config import ClArgsConfig
from mazes import load_dataset, Mazes, render_discrete
from models import BfsNCA, FixedBfsNCA, GCN, MLP, NCA
from models.nn import PathfindingNN
from training import train
from utils import Logger, VideoWriter, get_discrete_loss, get_mse_loss, to_path, load


np.set_printoptions(threshold=sys.maxsize)


def main_experiment(cfg=None):
    os.system('nvidia-smi -L')
    if torch.cuda.is_available():
            print('Using GPU/CUDA.')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
            print('Not using GPU/CUDA, using CPU.')
        
    if cfg is None:
        cfg = ClArgsConfig()
        cfg.set_exp_name()
    model_cls = globals()[cfg.model]

    print(f"Running experiment with config:\n {json.dumps(vars(cfg), indent=4)}")
    
    # Setup training
    model = model_cls(cfg)
    # Set a dummy initial maze state.
    model.reset(torch.zeros(cfg.minibatch_size, cfg.n_in_chan, cfg.width + 2, cfg.width + 2), is_torchinfo_dummy=True)

    dummy_input = model.seed(batch_size=cfg.minibatch_size)
    # FIXME: ad hoc (?)
    if cfg.model == "MLP":
        torchinfo.summary(model, input_size=dummy_input.shape)

    else:
        torchinfo.summary(model, input_size=dummy_input.shape)

    # param_n = sum(p.numel() for p in model.parameters())
    # print('model param count:', param_n)
    opt = torch.optim.Adam(model.parameters(), cfg.learning_rate)
    try:
        maze_data_train, maze_data_val, maze_data_test = load_dataset(cfg.n_data, cfg.device)
    except FileNotFoundError as e:
        print("No maze data files found. Run `python mazes.py` to generate the dataset.")
        raise
    maze_data_val.get_subset(cfg.n_val_data)
  # cfg.n_data = maze_data_train.mazes_onehot.shape[0]

    if cfg.load:
        model, opt, logger = load(model, opt, cfg)

        if cfg.test:
            return evaluate(model, maze_data_test, "test", cfg)

    else:
        if cfg.overwrite and os.path.exists(cfg.log_dir):
            shutil.rmtree(cfg.log_dir)
        try:
            os.mkdir(cfg.log_dir)
            logger = Logger()
        except FileExistsError as e:
            raise FileExistsError(f"Experiment log folder {cfg.log_dir} already exists. Use `--load` or `--overwrite` command line arguments "\
            "to load or overwrite it.")

    # Save a dictionary of the config to a json for future reference.
    json.dump(cfg.__dict__, open(f'{cfg.log_dir}/config.json', 'w'))

    mazes_onehot, mazes_discrete, maze_ims, target_paths = \
        (maze_data_train.mazes_onehot, maze_data_train.mazes_discrete, maze_data_train.maze_ims, maze_data_train.target_paths)

# fig, ax = plt.subplots(figsize=(20, 5))
# plt.imshow(np.hstack(maze_ims[:cfg.render_minibatch_size]))

    if cfg.model == "FixedBfsNCA":
        path_chan = 5
    else:
        path_chan = 4
        assert path_chan == mazes_discrete.max() + 1
# solved_mazes = mazes_discrete.clone()
# solved_mazes[torch.where(target_paths == 1)] = path_chan
# fig, ax = plt.subplots(figsize=(20, 5))
# plt.imshow(np.hstack(solved_maze_ims[:cfg.render_minibatch_size]))

# fig, ax = plt.subplots(figsize=(20, 5))
# plt.imshow(np.hstack(target_paths[:cfg.render_minibatch_size].cpu()))
# plt.tight_layout()

    assert cfg.n_in_chan == mazes_onehot.shape[1]

    # The set of initial mazes (padded with 0s, channel-wise, to act as the initial state of the CA).

    if cfg.render:
        render_trained(model, maze_data_train, cfg)
    else:
        train(model, opt, maze_data_train, maze_data_val, target_paths, logger, cfg)


def render_trained(model: PathfindingNN, maze_data, cfg, pyplot_animation=True):
    """Generate a video showing the behavior of the trained NCA on mazes from its training distribution.
    """
    mazes_onehot, mazes_discrete = maze_data.mazes_onehot, maze_data.mazes_discrete
    pool = model.seed(batch_size=mazes_onehot.shape[0])
    render_minibatch_size = min(cfg.render_minibatch_size, mazes_onehot.shape[0])
    batch_idx = np.random.choice(pool.shape[0], render_minibatch_size, replace=False)
    x = pool[batch_idx]
    x0 = mazes_onehot[batch_idx]
    model.reset(x0)

    if pyplot_animation:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))


        def get_imgs(x, x0, oracle_out=None):
            # x_img = to_path(x[:cfg.render_minibatch_size]).cpu()
            # x_img = (x_img - x_img.min()) / (x_img.max() - x_img.min())
            # x_img = np.hstack(x_img.detach())
            # solved_maze_ims = [render_discrete(x0i.argmax(0)[None,...]) for x0i in x0]
            # solved_mazes_im = np.hstack(np.vstack(solved_maze_ims))
            # vstackable_ims = [x_img]
            maze_imgs = np.hstack(render_discrete(mazes_discrete[batch_idx], cfg))
            vstackable_ims = []

            # Render some more arbitrary hidden channels
            for i in range(3):
                xi_img = x[:cfg.render_minibatch_size, -i-1].cpu()
                xi_img = (xi_img - xi_img.min()) / (xi_img.max() - xi_img.min())
                xi_img = np.hstack(xi_img.detach())
                vstackable_ims.append(xi_img)

            if oracle_out is not None:
                # Additionally render the evolution of the oracle model's age channel.
                img = oracle_out[:cfg.render_minibatch_size, -1, :, :, None]
                img = (img - img.min()) / (img.max() - img.min())
                img = np.hstack(img.cpu().detach())
                vstackable_ims.append(np.tile(img, (1, 1, 3)))
            bw_imgs = np.vstack(vstackable_ims)

            return maze_imgs, bw_imgs

        maze_imgs, bw_imgs = get_imgs(x, x0)
        global im1, im2
        im1 = ax1.imshow(maze_imgs)
        im2 = ax2.imshow(bw_imgs)
        plt.tight_layout()

        # def init():
        #     im1 = ax1.imshow(maze_imgs)
        #     im2 = ax2.imshow(bw_imgs)

        #     return im1, im2

        xs = []
        oracle_outs = []

        with torch.no_grad():
            xs.append(x)
            oracle_outs.append(model.oracle_out) if cfg.model == "BfsNCA" else None
            for i in range(cfg.n_layers):
                x = model(x)
                xs.append(x)
                oracle_outs.append(model.oracle_out) if oracle_outs else None


        # FIXME: We shouldn't need to call `imshow` from scratch here!!
        def animate(i):
            xi = xs[i]
            oracle_out = oracle_outs[i] if oracle_outs else None
            maze_imgs, bw_imgs = get_imgs(xi, x0, oracle_out=oracle_out)
            # ax1.imshow(maze_imgs)
            # im1.set_array(bw_imgs)
            ax2.imshow(bw_imgs)
            # im2.set_array(bw_imgs)
            # ax1.figure.canvas.draw()
            # ax2.figure.canvas.draw()

            return im2,


        anim = animation.FuncAnimation(
        fig, animate, interval=1, frames=cfg.n_layers, blit=True, save_count=50)
        anim.save(f'{cfg.log_dir}/path_nca_anim.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
            

def evaluate_train(model, cfg):
    """Evaluate the trained model on the training set."""
    # TODO: 
    pass


def evaluate(model, maze_data, name, cfg):
    """Evaluate the trained model on a test set."""
    n_eval_minibatches = 10
    # n_test_mazes = n_test_minibatches * cfg.minibatch_size
    test_mazes_onehot, test_mazes_discrete, target_paths = maze_data.mazes_onehot, maze_data.mazes_discrete, \
        maze_data.target_paths

    with torch.no_grad():
        eval_losses = []
        eval_discrete_losses = []
        i = 0

        for i in range(n_eval_minibatches):
            batch_idx = np.arange(i*cfg.minibatch_size, (i+1)*cfg.minibatch_size, dtype=int)
            x0 = test_mazes_onehot[batch_idx]
            x0_discrete = test_mazes_discrete[batch_idx]
            x = model.seed(batch_size=cfg.minibatch_size)
            target_paths_mini_batch = target_paths[batch_idx]
            model.reset(x0)

            for j in range(cfg.n_layers):
                x = model(x)

                if j == cfg.n_layers - 1:
                    test_loss = get_mse_loss(x, target_paths_mini_batch).item()
                    test_discrete_loss = get_discrete_loss(x, target_paths_mini_batch).item()
                    eval_losses.append(test_loss)
                    eval_discrete_losses.append(test_discrete_loss)
                    # clear_output(True)
                    fig, ax = plt.subplots(figsize=(10, 10))
                    solved_maze_ims = np.hstack(render_discrete(x0_discrete[:cfg.render_minibatch_size], cfg))
                    target_path_ims = np.tile(np.hstack(
                        target_paths_mini_batch[:cfg.render_minibatch_size].cpu())[...,None], (1, 1, 3)
                        )
                    predicted_path_ims = to_path(x[:cfg.render_minibatch_size])[...,None].cpu()
                    # img = (img - img.min()) / (img.max() - img.min())
                    predicted_path_ims = np.hstack(predicted_path_ims)
                    predicted_path_ims = np.tile(predicted_path_ims, (1, 1, 3))
                    imgs = np.vstack([solved_maze_ims, target_path_ims, predicted_path_ims])
                    # plt.imshow(imgs)
                    # plt.show()

    mean_eval_loss = np.mean(eval_losses)
    std_eval_loss = np.std(eval_losses)
    mean_discrete_eval_loss = np.mean(eval_discrete_losses)
    std_discrete_eval_loss = np.std(eval_discrete_losses)
    # print(f'Mean {name} loss: {mean_eval_loss}\nMean {name} discrete loss: {mean_discrete_eval_loss}') 

    # Dump stats to a json file
    stats = {
        'mean_eval_loss': mean_eval_loss,
        'std_eval_loss': std_eval_loss,
        'mean_discrete_eval_loss': mean_discrete_eval_loss,
        'std_discrete_eval_loss': std_discrete_eval_loss,
    }
    with open(f'{cfg.log_dir}/{name}_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print(json.dumps(stats, indent=4))

    return stats


if __name__ == '__main__':
    main_experiment()