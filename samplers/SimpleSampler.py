import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pdb
class SimpleSampler:
    def __init__(self, dataset, params):
      self.dataset = dataset
      self.total_size = dataset.__len__()
      self.batch_size = params["batch_size"]
      self.current_idx = 0
    def reset(self):
      self.current_idx = 0
    def next(self):
      xs = []
      ys = []
      for i in range(self.current_idx, min(self.current_idx + self.batch_size, self.total_size)):
          x,y = self.dataset[i]
          xs.append(x)
          ys.append(y)
      X = np.stack(xs)
      y = np.stack(ys)
      self.current_idx = self.current_idx + self.batch_size
      if self.dataset.regression:
          return X.shape[0], (torch.FloatTensor(X), torch.FloatTensor(y))
      else:
          return X.shape[0], (torch.FloatTensor(X), torch.LongTensor(y))
