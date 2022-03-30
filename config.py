

class Config():
  data_n = 256
  render_minibatch_size = 8  # how many mazes to render at once
  log_dir = 'log'
  step_n = 64  # Number of steps after which we calculate loss and update the network.
  expected_net_steps = 64 * 4  # How long an episode should be, in expectation. Random because we sample from pool of 
                               # in-progress mazes at random.
  n_aux_chan = 48  # The number in/out channels, including the input channels (n_in_chan)
  n_hidden_chan = 96  # The number of channels in the hidden layers.
  log_interval = 64  # How often to print results to the console.
  save_interval = 64  # How often to save the model and optimizer to disk.
  n_updates = 100000  # How many updates to perform during training.
  minibatch_size = 4  # How many mazes on which to train at once

