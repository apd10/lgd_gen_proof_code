import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr
import pdb
import torch

class L2LSH_TORCH():
  def __init__(self, params, num_hashes): 
    # N = number of hashes 
    # d = dimensionality
    # r = "bandwidth"
    self.N = num_hashes
    self.d = params["dimension"]
    self.r = params["bandwidth"]
    self.max_norm = None
    if "max_norm" in params:
        self.max_norm = params["max_norm"]
    self.device_id = params["device_id"]
    if "np_seed" in params:

        print("setting seed in l2lsh")
        np.random.seed(params["np_seed"])

    # set up the gaussian random projection vectors
    self.W = np.random.normal(size = (self.d, self.N))
    # normalize
    norms = np.sqrt(np.sum(np.multiply(self.W, self.W), axis=0)) #1 x N
    self.W = torch.FloatTensor(self.W / norms)
    self.b = torch.FloatTensor(np.random.uniform(low = 0,high = self.r,size = self.N))


  def compute(self, x):  # b x d
    values =   torch.floor((torch.matmul(x, self.W) + self.b)/ self.r ).long() # b x N
    return values
    

  def get_max(self):
    assert(self.max_norm is not None)
    return np.floor(self.max_norm / self.r)

  def get_min(self):
    assert(self.max_norm is not None)
    return np.floor(-self.max_norm / self.r)

  def get_dictionary(self):
    dic = {}
    dic["N"] =self.N
    dic["d"] =self.d
    dic["r"] =self.r
    dic["max_norm"] = self.max_norm
    dic["W"] = self.W.cpu()
    dic["b"] = self.b.cpu()
    return dic

  def set_dictionary(self, dic):
    self.N = dic["N"]
    self.d = dic["d"]
    self.r = dic["r"]
    self.max_norm = dic["max_norm"]
    self.W = dic["W"]
    self.b = dic["b"]
    if self.device_id != -1:
      self.W = self.W.cuda(self.device_id)
      self.b = self.b.cuda(self.device_id)


  def get_w(self, rep, chunk_size):
    Worig = np.array(self.W.cpu())
    W = Worig.transpose()[rep*chunk_size:(rep+1)*chunk_size,:]
    return W

  def get_b(self, rep, chunk_size):
    Borig = np.array(self.b.cpu())
    B = Borig[rep*chunk_size:(rep+1)*chunk_size]
    return B
  
  def get_r(self):
    return np.float(self.r)

  def get_equations(self, hashvalues, rep, chunk_size):
    assert(chunk_size == len(hashvalues)) # remove chunk_size
    Worig = np.array(self.W.cpu())
    Borig = np.array(self.b.cpu())

    W = Worig.transpose()[rep*chunk_size:(rep+1)*chunk_size,:]
    B = Borig[rep*chunk_size:(rep+1)*chunk_size]
    W_heq_1 = np.copy(W) # W_heq : N x d 
    b_heq_1 = np.zeros(chunk_size)
    # lower bound
    # (Wx + b) / r >= hashvalue
    #  Wx >= r*hashvalue - b
    # -Wx <= (r*hashvalue - b) * -1
    for i in range(chunk_size):
        W_heq_1[i, :] = W_heq_1[i, :] * -1
        b_heq_1[i] = (hashvalues[i] * self.r  - B[i]) * -1

    W = Worig.transpose()[rep*chunk_size:(rep+1)*chunk_size,:]
    B = Borig[rep*chunk_size:(rep+1)*chunk_size]
    W_heq_2 = np.copy(W) # W_heq : N x d 
    b_heq_2 = np.zeros(chunk_size)
    # upper bound
    # (Wx + b) / r < (hashvalue+1)
    #  Wx < r*hashvalue - b
    for i in range(chunk_size):
        b_heq_2[i] = (hashvalues[i]+1) * self.r  - B[i]
    W_heq = np.concatenate([W_heq_1, W_heq_2])
    b_heq = np.concatenate([b_heq_1, b_heq_2])
    return W_heq, b_heq
    
  def score_sample():
    ''' this will compute the sum of scores of all the samples according to the l2lsh 
        kernel density
    '''
