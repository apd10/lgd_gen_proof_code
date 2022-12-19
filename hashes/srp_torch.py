import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr
import pdb
import torch

class SRP_TORCH():
  def __init__(self, params, num_hashes, num_bits=16): 
    # N = number of hashes 
    # d = dimensionality
    self.N = num_hashes
    self.num_bits = num_bits
    self.d = params["dimension"]
    self.seed = params["seed"]



    np.random.seed(self.seed)
    W = np.random.normal(size = (self.d, self.N, self.num_bits)) # D x num_hashes x num_bits
    powersOfTwo =  np.array([2**i for i in range(self.num_bits)])
    self.W = torch.FloatTensor(W)
    self.powersOfTwo = torch.FloatTensor(powersOfTwo)
    self.device_id = params["device_id"]
    if self.device_id != -1:
      self.W = self.W.cuda(self.device_id)
      self.powersOfTwo = self.powersOfTwo.cuda(self.device_id)
    
    
  def hash(self,x):  # B x d
    values = torch.tensordot(x,self.W,dims=([-1],[0])) >=0 
    values = values.type(torch.float32) 
    values =   torch.matmul(values, self.powersOfTwo) # OR:  torch.tensordot(values,self.powersOfTwo,dims([-1],[0]))
    values = values.type(torch.int32)
    return values
    
  
