
import base64
import io
import pickle
import requests

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
# os.environ['FFMPEG_BINARY'] = 'ffmpeg'
# import moviepy.editor as mvp
import numpy as np
import PIL
import torch


class Logger():
  def __init__(self):
    self.loss_log = []

  def log(self, loss):
    self.loss_log.append(loss)

ca_state_fname = 'ca_state_dict.pt'
opt_state_fname = 'opt_state_dict.pt'
logger_fname = 'logger.pk'
maze_data_fname = 'maze_data.pk'

def save(ca, opt, maze_data, logger, cfg):
  torch.save(ca.state_dict(), f'{cfg.log_dir}/{ca_state_fname}')
  torch.save(opt.state_dict(), f'{cfg.log_dir}/{opt_state_fname}')
  with open(f'{cfg.log_dir}/{logger_fname}', 'wb') as f:
    pickle.dump(logger, f)
  with open(f'{cfg.log_dir}/{maze_data_fname}', 'wb') as f:
    pickle.dump(maze_data, f)


def load(ca, opt, cfg):
  ca.load_state_dict(torch.load(f'{cfg.log_dir}/{ca_state_fname}'))
  opt.load_state_dict(torch.load(f'{cfg.log_dir}/{opt_state_fname}'))
  maze_data = pickle.load(open(f'{cfg.log_dir}/{maze_data_fname}', 'rb'))
  logger = pickle.load(open(f'{cfg.log_dir}/{logger_fname}', 'rb'))
  print(f'Loaded CA and optimizer state dict, and maze archive from {cfg.log_dir}.')

  return ca, opt, maze_data, logger


def to_path(x):
  return x[:, -1, :, :]


def get_mse_loss(x, target_paths):
  paths = to_path(x)
  # overflow_loss = (x-x.clamp(-1.0, 1.0)).abs().sum()
  err = paths - target_paths
  # err = (paths - paths.min(dim)) / (paths.max() - paths.min()) - path_targets
  loss = err.square().mean()  #+ overflow_loss
  return loss


def gen_pool(size, n_chan, height, width):
  # A set of mazes-in-progress, which will allow us to effectively run episodes up to length m * step_m (where m is some 
  # integer), while computing loss against optimal path and updating weights every step_n-many steps.
  pool = torch.zeros(size, n_chan, height, width)

  return pool


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
#   display(Image(data=imencode(a, fmt)))


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