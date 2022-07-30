from pdb import set_trace as TT
import math
import os
import shutil
import PIL
import cv2
import imageio
from matplotlib import animation, pyplot as plt
import numpy as np
import torch as th
from mazes import Tiles, render_discrete

from models.nn import PathfindingNN


RENDER_TYPE = 0
N_RENDER_CHANS = 1
RENDER_BORDER = False
SAVE_GIF = True
SAVE_PNGS = True
N_RENDER_EPISODES = 10
CV2_WAIT_KEY_TIME = 1
RENDER_WEIGHTS = False
CV2_IM_SIZE = (1500, 1500)


def render_trained(model: PathfindingNN, maze_data, cfg, pyplot_animation=True, name=''):
    """Generate a video showing the behavior of the trained NCA on mazes from its training distribution.
    """
    model.eval()
    mazes_onehot, mazes_discrete, edges, target_paths = maze_data.mazes_onehot, maze_data.mazes_discrete, maze_data.edges,\
         maze_data.target_paths
    if N_RENDER_CHANS is None:
        n_render_chans = model.n_out_chan
    else:
        n_render_chans = min(N_RENDER_CHANS, model.n_out_chan)

    render_minibatch_size = min(cfg.render_minibatch_size, mazes_onehot.shape[0])
    path_lengths = target_paths.sum((1, 2))
    path_chan = mazes_discrete[0].max() + 1
    mazes_discrete = th.where((mazes_discrete == Tiles.EMPTY) & (target_paths == 1), path_chan, mazes_discrete)

    # Render most complex mazes first
    batch_idxs = path_lengths.sort(descending=True)[1]
    # batch_idxs = th.arange(mazes_onehot.shape[0])

    batch_idx = np.random.choice(mazes_onehot.shape[0], render_minibatch_size, replace=False)
    bi = 0

    global N_RENDER_EPISODES
    if N_RENDER_EPISODES is None:
        N_RENDER_EPISODES = len(batch_idxs)

    def reset(bi):
        batch_idx = batch_idxs[th.arange(render_minibatch_size) + bi]
        pool = model.seed(batch_size=mazes_onehot.shape[0], width=cfg.width, height=cfg.height,)
        x = pool[batch_idx]
        x0 = mazes_onehot[batch_idx]
        model.reset(x0, e0=edges, new_batch_size=True)
        maze_imgs = np.hstack(render_discrete(mazes_discrete[batch_idx], cfg))
        bi = (bi + render_minibatch_size) % mazes_onehot.shape[0]

        return x, maze_imgs, bi


    def get_imgs(x, oracle_out=None, maze_imgs=None):
        # x_img = to_path(x[:cfg.render_minibatch_size]).cpu()
        # x_img = (x_img - x_img.min()) / (x_img.max() - x_img.min())
        # x_img = np.hstack(x_img.detach())
        # solved_maze_ims = [render_discrete(x0i.argmax(0)[None,...]) for x0i in x0]
        # solved_mazes_im = np.hstack(np.vstack(solved_maze_ims))
        # vstackable_ims = [x_img]
        stackable_ims = []

        # Render the path channel and some other arbitrary hidden channels
        for i in range(n_render_chans):
            xi_img = x[:cfg.render_minibatch_size, -i-1].cpu()
            xi_img = (xi_img - xi_img.min()) / (xi_img.max() - xi_img.min())
            xi_img = np.hstack(xi_img.detach())
            stackable_ims.append(xi_img)

        if oracle_out is not None:
            # Additionally render the evolution of the oracle model's age channel.
            img = oracle_out[:cfg.render_minibatch_size, -1, :, :, None]
            img = (img - img.min()) / (img.max() - img.min())
            img = np.hstack(img.cpu().detach())
            stackable_ims.append(np.tile(img, (1, 1, 3)))
        
        if n_render_chans > 3:
            n_cols = math.ceil(math.sqrt(n_render_chans + (1 if maze_imgs is not None else 0)))
        else:
            n_cols = n_render_chans + (1 if maze_imgs is not None else 0)

        # Give b/w images a third channel, ade maze as first tile
        if maze_imgs is not None:
            stackable_ims = [maze_imgs] + [np.tile(im[...,None], (1, 1, 3)) for im in stackable_ims]
            vert_bar = np.zeros((stackable_ims[-1].shape[0], 1, 3))
            # hor_bar = np.zeros((1, stackable_ims[-1].shape[0], 3))
        else:
            vert_bar = np.zeros((stackable_ims[0].shape[0], 1))
            # hor_bar = np.zeros((1, stackable_ims[0].shape[0]))

        ims = [np.concatenate([np.concatenate((im, vert_bar), axis=1) for im in stackable_ims[i:i+n_cols]], axis=1) \
            for i in range(0, len(stackable_ims), n_cols)]

        # Pad the last row with empty squares
        if len(ims) > 1:
            last_row = np.zeros((ims[-2].shape[0], *ims[-2].shape[1:]))
            last_row[:, :ims[-1].shape[1]] = ims[-1]
            ims[-1] = last_row

        hor_bar = np.zeros((1, *ims[0].shape[1:]))
        ims = np.concatenate([hor_bar] + [np.concatenate((row_im, hor_bar), axis=0) for row_im in ims], axis=0)

        width = stackable_ims[0].shape[0]
        height = stackable_ims[0].shape[1]
        
        # Highlight the path channel (assuming it is the second tile in our grid of rendered channels).
        if maze_imgs is not None and RENDER_BORDER:
            ims[:width+1, height, 2] = 1
            ims[:width+1, 2*height+1, 2] = 1
            ims[0, height:2*height+2, 2] = 1
            ims[width+1, height:2*height+2, 2] = 1

        return ims

    model_has_oracle = model == "BfsNCA"

    if RENDER_WEIGHTS:
        for name, p in model.named_parameters():
            if "weight" in name:
                # (in_chan, height, width, out_chan)
                im = p.data.permute(0, 2, 3, 1)
                # (in_chan * height, width, out_chan)
                im = th.vstack([wi for wi in im])
                # (out_chan, width, in_chan * height)
                im = im.permute(2, 1, 0)
                im = th.vstack([wi for wi in im])
                im = (im - im.min()) / (im.max() - im.min())
                im = PIL.Image.fromarray(im.cpu().numpy() * 255)
                im.show()
                im.save(open(os.path.join(cfg.log_dir, 'weights.png')))
        return

    # Render live and indefinitely using cv2.
    if RENDER_TYPE == 0:

        def render_loop(bi, writer=None):
            # Create large window
            # cv2.namedWindow('maze', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('model', cv2.WINDOW_NORMAL)
            cv2.namedWindow('pathfinding', cv2.WINDOW_NORMAL)
            render_dir = os.path.join(cfg.log_dir, f"render{name}")
            if SAVE_PNGS:
                if os.path.exists(render_dir):
                    shutil.rmtree(render_dir)
                os.mkdir(render_dir)
            # cv2.resize('pathfinding', (2000, 2000))
            frame_i = 0

            for ep_i in range(N_RENDER_EPISODES):
                x, maze_imgs, bi = reset(bi)
                oracle_out = model.oracle_out if model_has_oracle else None
                ims = get_imgs(x, oracle_out=oracle_out, maze_imgs=maze_imgs)
                imS = cv2.resize(ims, CV2_IM_SIZE)
                cv2.imshow('pathfinding', imS)
                # Save image as png
                if SAVE_PNGS:
                    cv2.imwrite(os.path.join(render_dir, f"render_{frame_i}.png"), ims)
                if writer:
                    writer.append_data(ims)

                # cv2.imshow('maze', maze_imgs)
                # cv2.imshow('model', get_imgs(x, oracle_out=oracle_out))
                cv2.waitKey(CV2_WAIT_KEY_TIME)
                for i in range(cfg.n_layers):
                    frame_i += 1
                    x = model.forward(x)
                    oracle_out = model.oracle_out if model_has_oracle else None
                    # cv2.imshow('model', get_imgs(x, oracle_out=oracle_out))
                    ims = get_imgs(x, oracle_out=oracle_out, maze_imgs=maze_imgs)
                    imS = cv2.resize(ims, CV2_IM_SIZE)
                    cv2.imshow('pathfinding', imS)
                    if SAVE_PNGS:
                        cv2.imwrite(os.path.join(render_dir, f"render_{frame_i}.png"), ims*255)
                    if writer:
                        writer.append_data(ims)
                    cv2.waitKey(CV2_WAIT_KEY_TIME)
            
        if SAVE_GIF:
            with imageio.get_writer(os.path.join(cfg.log_dir, f"behavior{name}.gif"), mode='I', duration=0.1) as writer:
                render_loop(bi, writer)
        else:
            render_loop(bi)


    # Save a video with pyplot.
    elif RENDER_TYPE == 1:
        plt.axis('off')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10), gridspec_kw={'width_ratios': [1, 10]})
        

        for i in range(N_RENDER_EPISODES):
            x, maze_imgs, bi = reset(bi)
            bw_imgs = get_imgs(x)
            global im1, im2
            mat1 = ax1.matshow(maze_imgs)
            mat2 = ax2.matshow(bw_imgs)
            plt.tight_layout()

            # def init():
            #     im1 = ax1.imshow(maze_imgs)
            #     im2 = ax2.imshow(bw_imgs)

            #     return im1, im2

            xs = []
            oracle_outs = []

            xs.append(x)
            oracle_outs.append(model.oracle_out) if model_has_oracle else None

            for j in range(cfg.n_layers):
                x = model.forward(x)
                xs.append(x)
                oracle_outs.append(model.oracle_out) if oracle_outs else None


        # FIXME: We shouldn't need to call `imshow` from scratch here!!
        def animate(i):
            xi = xs[i]
            oracle_out = oracle_outs[i] if oracle_outs else None
            bw_imgs = get_imgs(xi, oracle_out=oracle_out)
            if i % cfg.n_layers == 0:
                # ax1.imshow(maze_imgs)
                mat1.set_array(bw_imgs)
            # ax2.imshow(bw_imgs)
            mat2.set_array(bw_imgs)
            # ax1.figure.canvas.draw()
            # ax2.figure.canvas.draw()

            return mat1, mat2,


        anim = animation.FuncAnimation(
        fig, animate, interval=1, frames=cfg.n_layers, blit=True, save_count=50)
        anim.save(f'{cfg.log_dir}/path_nca_anim{name}.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

