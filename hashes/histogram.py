import numpy as np
from scipy.stats import norm # for P_L2
import math 
from scipy.special import ndtr
import pdb
import torch

class HISTOGRAM():
  def __init__(self, params): 
    # d = dimensionality
    # r = "bandwidth"
    self.d = params["dimension"]
    self.r = params["bandwidth"]


  def compute(self, x):  # b x d
    values =   torch.floor( x / self.r ).long() # b x d # we just do a normal histogram of width self.r
    return values
    

  def get_max(self):
    assert(False)
    return None

  def get_min(self):
    assert(False)
    return None

  def get_dictionary(self):
    dic = {}
    dic["d"] =self.d
    dic["r"] =self.r
    return dic

  def set_dictionary(self, dic):
    self.d = dic["d"]
    self.r = dic["r"]


  def get_w(self, rep, chunk_size):
    assert(False)
    return None

  def get_b(self, rep, chunk_size):
    assert(False)
    return None
  
  def get_r(self):
    return np.float(self.r)

  def get_equations(self, hashvalues, rep, chunk_size):
    assert(False)
    return None
    
  def score_sample():
    ''' this will compute the sum of scores of all the samples according to the l2lsh 
        kernel density
    '''
