import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pdb
from dataparsers.CSVParser import CSVParser


def compute_next_batch(batch, dataset_norm, dataset_full, batch_norm, size, label, CHUNK=32):
    indices = torch.argsort(batch_norm, descending=True)
    indices = indices[:10*CHUNK][torch.randperm(10*CHUNK)[:CHUNK]]
    batch = batch[indices] 
    batch = batch / torch.norm(batch, dim=1).reshape(-1,1)
    cosine_sim = torch.mm(batch, dataset_norm.T)
    new_indices = torch.argsort(cosine_sim, dim=1, descending=True)[:,10*CHUNK:20*CHUNK]
    uniques = torch.unique(new_indices.reshape(-1))
    uniques = uniques[torch.randperm(len(uniques))[:size]]
    return dataset_full[uniques], torch.ones(len(uniques)) * label;



class LGDExactSampler:
    def __init__(self, dataset, params):
      self.dataset = dataset
      self.total_size = dataset.__len__()
      self.batch_size = params["batch_size"]
      self.nn = int(params["nn_percent"] * self.batch_size)
      self.reset_itr = int(params["nn_reset"])
      self.current_idx = 0
      self.itr = 0
      if type(dataset) == CSVParser: # assumes classification
          self.label_ids = np.unique(dataset.labels)
          self.complete_data = dataset.X.to_numpy()[:,:8] # TODO
          self.complete_labels = dataset.labels
          self.label_specific_data_full = {}
          self.label_specific_data_norm = {}
          for l in self.label_ids:
              self.label_specific_data_full[l] = torch.from_numpy(self.complete_data[self.complete_labels == l]).float().cuda()
              self.label_specific_data_norm[l] = self.label_specific_data_full[l] / torch.norm(self.label_specific_data_full[l], dim=1).reshape(-1,1)
      else:
          raise NotImplementedError # can do general data generation

      self.payload = None

    def reset(self):
      self.current_idx = 0
      self.itr = 0

    def next(self):
      xs = []
      ys = []

      # random samples
      self.itr = self.itr + 1
      if (self.reset_itr != -1 and (self.itr % self.reset_itr == 0)) or self.payload is None:
          random_batch = self.batch_size
      else:
          random_batch = self.batch_size - self.nn

      for i in range(self.current_idx, min(self.current_idx + random_batch, self.total_size)):
          x,y = self.dataset[i]
          xs.append(x)
          ys.append(y)

      X = np.stack(xs)
      y = np.stack(ys)
      self.current_idx = self.current_idx + random_batch

      # nn samples
      if self.payload is not None:
          oldx, oldy, ft_per_sample_grads = self.payload
          batch_norm = torch.zeros(oldx.size(0), device=oldx.device)
          for item in ft_per_sample_grads:
              batch_norm +=  torch.linalg.norm(item, dim=tuple(range(1,len(item.shape))))
          for label in self.label_ids:
              xnn, ynn = compute_next_batch(oldx[oldy==label], self.label_specific_data_norm[label], self.label_specific_data_full[label],
                                            batch_norm[oldy==label], int(self.nn / len(self.label_ids)), label)
              X = np.concatenate([X, xnn.cpu().numpy()], axis=0)
              y = np.concatenate([y, ynn.cpu().numpy()], axis=0)
        
      if self.dataset.regression:
          return X.shape[0], (torch.FloatTensor(X), torch.FloatTensor(y))
      else:
          return X.shape[0], (torch.FloatTensor(X), torch.LongTensor(y))
