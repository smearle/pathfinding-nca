import base64
import io
import json
import os
from pdb import set_trace as TT
import pickle
import shutil
import requests

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
# os.environ['FFMPEG_BINARY'] = 'ffmpeg'
# import moviepy.editor as mvp
import numpy as np
import PIL
import torch as th
from configs.config import Config
from mazes import Tiles
from models.gnn import GCN
from render import N_RENDER_CHANS, render_ep_cv2

from models.nn import PathfindingNN


corner_idxs_3x3 = th.Tensor([
    [0, 0],
    [0, -1],
    [-1, 0],
    [-1, -1],
]).long()

# Here we cut the corners and the tiles directly adjacent to the corner. Our hand-coded DFS doesn't use these tiles.
# This is the most tiles we can exclude while keeping the weight symmetric.
corner_idxs_5x5 = th.Tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [0, -1],
    [0, -2],
    [1, -1],
    [-1, 0],
    [-2, 0],
    [-1, 1],
    [-1, -1],
    [-2, -1],
    [-1, -2],
]).long()


class Logger():
    def __init__(self):
        self.loss_log = []
        self.discrete_loss_log = []
        self.val_stats_log = {}
        self.n_step = 0

    def log(self, loss, discrete_loss):
        """Log training loss (once per update step)."""
        self.loss_log.append(loss)
        self.discrete_loss_log.append(discrete_loss)
        self.n_step += 1

    def log_val(self, val_stats):
        """Log validation loss."""
        self.val_stats_log[self.n_step] = val_stats

    def get_val_stat(self, k):
        return {i: self.val_stats_log[i][k] for i in self.val_stats_log}

ca_state_fname = 'ca_state_dict.pt'
opt_state_fname = 'opt_state_dict.pt'
logger_fname = 'logger.pk'


def backup_file(fname):
    if os.path.isfile(fname):
        shutil.copyfile(fname, fname + '.bkp')


def delete_backup(fname):
    if os.path.isfile(fname + '.bkp'):
        os.remove(fname + '.bkp')


def save(ca, opt, logger, cfg):
    model_path = f'{cfg.log_dir}/{ca_state_fname}'
    optimizer_path = f'{cfg.log_dir}/{opt_state_fname}'
    backup_file(model_path)
    backup_file(optimizer_path)
    th.save(ca.state_dict(), model_path)
    th.save(opt.state_dict(), optimizer_path)
    delete_backup(model_path)
    delete_backup(optimizer_path)
    with open(f'{cfg.log_dir}/{logger_fname}', 'wb') as f:
        pickle.dump(logger, f)


def load(model, opt, cfg):
    if not th.cuda.is_available():
        map_location = th.device('cpu')
    else:
        map_location = None
    try:
        model.load_state_dict(th.load(f'{cfg.log_dir}/{ca_state_fname}', map_location=map_location))
        opt.load_state_dict(th.load(f'{cfg.log_dir}/{opt_state_fname}', map_location=map_location))
    except Exception:  #FIXME: lol
        model.load_state_dict(th.load(f'{cfg.log_dir}/{opt_state_fname}', map_location=map_location))
        opt.load_state_dict(th.load(f'{cfg.log_dir}/{ca_state_fname}', map_location=map_location))
    logger = pickle.load(open(f'{cfg.log_dir}/{logger_fname}', 'rb'))
    print(f'Loaded CA and optimizer state dict, maze archive, and logger from {cfg.log_dir}.')


    return model, opt, logger


def log_stats(model: PathfindingNN, logger: Logger, cfg: Config):
    n_params = count_parameters(model, cfg)
    with open(f"{cfg.log_dir}/stats.json", "w") as f:
        json.dump({
            "n_params": n_params,
            "n_updates": logger.n_step,
        }, f, indent=4)


def count_parameters(model: PathfindingNN, cfg: Config):
    n_params = 0
    for name, p in model.named_parameters():
        if "weight" in name:
            if not isinstance(model, GCN) and cfg.cut_conv_corners:
                # Don't count the corners.
                if p.shape[-2:] == (3, 3):
                    n_ps = p.numel() * (9 - corner_idxs_3x3.shape[0])/9
                elif p.shape[-2:] == (5, 5):
                    n_ps = p.numel() * (25 - corner_idxs_5x5.shape[0])/25
                assert n_ps % 1 == 0
                n_params += int(n_ps)
            elif cfg.symmetric_conv:
                assert p.shape[-2:] == (3, 3)
                n_ps = p.numel() * 2/9
                assert n_ps % 1 == 0
                n_params += int(n_ps)
            else:
                n_params += p.numel()

        # Some models have a mix of learnable weights and a handcoded "oracle", with potentially frozen weights.
        # if ("Bfs" in cfg.model or "Dfs" in cfg.model) and "oracle" in name:
            # assert not p.requires_grad
        # else:
            # assert p.requires_grad

    return n_params


def to_path(x):
    return x[:, -1, :, :]


def get_mse_loss(paths, target_paths, cfg=None):
    # Assuming dimension 0 is batch dimension.
    err = (paths - target_paths).square().mean(dim=(1, 2))
    return err


def get_discrete_loss(x, target_paths, cfg=None):
    paths = to_path(x).round()
    err = paths - target_paths
    loss = err.square()  #.float().mean()    #+ overflow_loss
    return loss


def solnfree_pathfinding_loss(paths, target_paths, hillclimbing_model):
    x = th.zeros((paths.shape[0], hillclimbing_model.n_in_chan, paths.shape[1], paths.shape[2]), device=paths.device)
    x[:, Tiles.WALL] = paths
    x[:, Tiles.SRC, 0, 0] = 1
    x[:, Tiles.SRC, -1, -1] = 1
    hillclimbing_model.reset(initial_maze=x)
    x = th.zeros((paths.shape[0], hillclimbing_model.n_hid_chan, paths.shape[1], paths.shape[2]), device=paths.device)
    for i in range(64):
        x = hillclimbing_model(x)
    return x[:, hillclimbing_model.path_chan, :, :].sum(dim=(-1, -2))


def test_maze_gen_loss(wall_empty, target_paths, pathfinding_mode, cfg):
    # fuzziness = th.mean(th.abs(wall_empty - 0.5))
    fuzziness = th.mean(th.abs(0.5 - th.abs(wall_empty - 0.5)))
    print('fuzziness', fuzziness.item())
    fuzziness_loss = fuzziness
    # return 1 - th.mean(th.abs(wall_empty - 0.5))
    # return fuzziness_loss
    pct_wall = th.mean(wall_empty)
    print('pct_wall', pct_wall.item())
    pct_wall_loss = abs(.5 - pct_wall)
    # return pct_wall_loss
    return 1 * fuzziness_loss + 2.0 * pct_wall_loss


def maze_gen_loss(wall_empty, target_paths, pathfinding_model, cfg):
    # return fuzziness_pct_wall_loss

    # FIXME: Hard-coding src/trg for now. Should probably get these from initial maze instead?
    x0 = th.zeros((wall_empty.shape[0], pathfinding_model.n_in_chan, wall_empty.shape[1], wall_empty.shape[2]), device=wall_empty.device)
    # wall_empty = th.tanh(wall_empty)
    wall_empty = th.stack([wall_empty, 1-wall_empty], dim=1)
    # Take the softmax over empty/wall to encourage near-binary output here.
    # wall_empty = th.softmax(3*wall_empty, dim=1)
    print('min', wall_empty.min().item(), 'max', wall_empty.max().item())
    fuzziness_pct_wall_loss = test_maze_gen_loss(wall_empty[:, 1], target_paths, pathfinding_model, cfg)
    return fuzziness_pct_wall_loss
    x0[:, (Tiles.WALL, Tiles.EMPTY)] = wall_empty
    x0[:, (Tiles.SRC, Tiles.EMPTY), 4, 4] = 1
    x0[:, Tiles.WALL, 4, 4] = 0
    x0[:, (Tiles.TRG, Tiles.EMPTY), 8, 8] = 1
    x0[:, Tiles.WALL, 8, 8] = 0
    pathfinding_model.reset(initial_maze=x0)
    x = th.zeros((wall_empty.shape[0], pathfinding_model.n_hid_chan, wall_empty.shape[-2], wall_empty.shape[-1]), device=wall_empty.device)
    # NOTE: This limits amount of pathfinding we can do to find paths of ~32. Is there a better way, other than increasing
    #   the number of iterations?
    if cfg.render:
        render_ep_cv2(0, pathfinding_model, mazes_onehot=x0, target_paths=target_paths, cfg=cfg, batch_idxs=np.arange(x0.shape[0]),
            n_render_chans=N_RENDER_CHANS, render_dir=None, model_has_oracle=False, frame_i=0, window_name='aux model')
    for i in range(64):
        x = pathfinding_model(x)
    sol_path = x[:, :, :, :]
    mean_path_len = th.mean(sol_path.sum(dim=(-2, -1)))
    trg_path_len = 100
    print('mean_path_len', mean_path_len.item())
    # Note that this punishes over- and under-long paths.
    path_len_loss = abs(trg_path_len - mean_path_len) / trg_path_len 
    return 1.2 * path_len_loss + fuzziness_pct_wall_loss


def imread(url, max_size=None, mode=None):
    if url.startswith(('http:', 'https:')):
        # wikimedia requires a user agent
        headers = {
            "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = PIL.Image.open(f)
    if max_size is not None:
        img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    if mode is not None:
        img = img.convert(mode)
    img = np.float32(img)/255.0
    return img

def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1)*255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt='jpeg'):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = 'png'
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def im2url(a, fmt='jpeg'):
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode('ascii')
    return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string


# def imshow(a, fmt='jpeg'):
#     display(Image(data=imencode(a, fmt)))


def tile2d(a, w=None):
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w-len(a))%w
    a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
    h = len(a)//w
    a = a.reshape([h, w]+list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
    return a


class VideoWriter:
    def __init__(self, filename='_autoplay.mp4', fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.params['filename'] == '_autoplay.mp4':
            self.show()

    def show(self, **kw):
            self.close()
            fn = self.params['filename']
            # display(mvp.ipython_display(fn, **kw))