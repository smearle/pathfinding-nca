import torch


class CA(torch.nn.Module):
  def __init__(self, n_in_chan=4, n_aux_chan=48, n_hidden_chan=96):
    super().__init__()
    self.n_out_chan = n_aux_chan - n_in_chan  # Writable channels the the output. (None-maze channels at the input.)
    self.w1 = torch.nn.Conv2d(n_aux_chan, n_hidden_chan, 3, padding=1)

    # Since the NCA receives the onehot maze as input at each step, we do not write to the channels reserved for this.
    self.w2 = torch.nn.Conv2d(n_hidden_chan, n_aux_chan - n_in_chan, 1) 

    self.w2.weight.data.zero_()

    # The initial/actual state of the board (so that we can use walls to simulate path-flood).
    self.x0 = None

  def forward(self, x, update_rate=0.5):
    y = torch.cat([self.x0, x], dim=1)
    y = self.w2(torch.relu(self.w1(y)))
    y = (torch.sigmoid(y) - 0.5) * 2
    b, c, h, w = y.shape
    return y

# def seed(self, n, sz=16):
#   return torch.zeros(n, self.chn, sz, sz)

  def reset(self, x0):
    self.x0 = x0

# def to_rgb(x):
  # return x[...,:3,:,:]+0.5
