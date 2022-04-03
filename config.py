import argparse
from pdb import set_trace as TT
import torch


class Config():
  """Default configuration."""
  # How many mazes on which to train at a time
  data_n = 256

  render_minibatch_size = 8  # how many mazes to render at once
  log_dir = 'log'

  # Number of steps after which we calculate loss and update the network.
  step_n = 64  

  # How long an episode should be, in expectation. Random because we sample from pool of in-progress mazes at random.  
  expected_net_steps = 64 * 4  

  # The number in/out channels, including the input channels (n_in_chan)
  n_aux_chan = 48  

  # The number of channels in the hidden layers.
  n_hidden_chan = 96  

  # How often to print results to the console.
  log_interval = 512  

  # How often to save the model and optimizer to disk.
  save_interval = log_interval * 1  

  # How many updates to perform during training.
  n_updates = 100000    

  # How many mazes on which to train at once
  minibatch_size = 32    

  # Size of minibatch for rendering images and animations.
  render_minibatch_size = 8

  # device = 'cuda' if torch.cuda.is_available() else 'cpu'

  learning_rate = 1e-4

  # Train on new random mazes. If None, train only on the initial set of generated mazes. 
  gen_new_data_interval = None


class ClArgsConfig():

  def __init__(self):
    """Set default arguments and get command line arguments."""
    # Load default arguments.
    cfg = Config()
    [setattr(self, k, getattr(cfg, k)) for k in dir(cfg) if not k.startswith('_')]

    # Get command-line arguments.
    args = argparse.ArgumentParser()
    args.add_argument('--load', action='store_true')
    args.add_argument('--render', action='store_true')

    # Include all parameters in the Config class as command-line arguments.
    [args.add_argument('--' + k, type=type(v), default=v) for k, v in self.__dict__.items()]
    args = args.parse_args()
    args.load = True if args.render else args.load

    # Command-line arguments will overwrite default config attributes.
    [setattr(self, k, v) for k, v in vars(args).items()]