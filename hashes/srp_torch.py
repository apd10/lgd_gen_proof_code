import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr
import pdb
import torch

class SRP_TORCH():
  def __init__(self, params, num_hashes): 
    # N = number of hashes 
    # d = dimensionality
    self.N = num_hashes
    self.d = params["dimension"]
    # set up the gaussian random projection vectors

    # combine params
    self.combine_mode = None
    if "combine_params" in params:
        cp = params["combine_params"]
        self.combine_mode = cp["combine_mode"]
        self.combine_params = cp[self.combine_mode]
        if self.combine_mode == "OR":
            self.N = self.N * cp[self.combine_mode]["l"]

    if "np_seed" in params:
        print("setting seed in srp")
        np.random.seed(params["np_seed"])
    W = np.random.normal(size = (self.d, self.N)) # D x num_hashes
    self.W = torch.FloatTensor(W)
    self.device_id = params["device_id"]
    if self.device_id != -1:
      self.W = self.W.cuda(self.device_id)




  def combine(self, values):
      if self.combine_mode == "OR":
          l = self.combine_params["l"]
          values = torch.any(values.reshape(values.shape[0], l, -1), dim=1)
      else:
          raise NotImplementedError
      return values


  def compute(self,x):  # b x d
    values =   torch.matmul(x, self.W) >= 0
    if self.combine_mode is not None:
        values = self.combine(values)
    values = values.type(torch.int32)
    return values
    

  def get_max(self):
    return 1

  def get_min(self):
    return 0

  def get_dictionary(self):
    dic = {}
    dic["N"] =self.N
    dic["d"] =self.d
    dic["W"] = self.W
    return dic

  def set_dictionary(self, dic):
    self.N = dic["N"]
    self.d = dic["d"]
    self.W = dic["W"]
    if self.device_id != -1:
        self.W = self.W.cuda(self.device_id)

  def get_equations(self, hashvalues, rep, chunk_size):
    assert(chunk_size == len(hashvalues)) # remove chunk_size
    W = np.array(self.W.cpu())

    W_heq = np.copy(W.transpose()[rep*chunk_size:(rep+1)*chunk_size,:]) # W_heq : N x d 
    b_heq = np.zeros(chunk_size)
    for i in range(chunk_size):
      if hashvalues[i] == 1:
          W_heq[i, :] = W_heq[i, :] * -1
    return W_heq, b_heq
    
  
