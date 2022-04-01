

import torch


class Config():
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

  # device = 'cuda' if torch.cuda.is_available() else 'cpu'

  learning_rate = 1e-4

  # Train on new random mazes 
  gen_new_data_interval = None

