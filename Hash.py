import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from hashes.list_hashfunctions import *

class HashFunction:
  def get(params, num_hashes, num_bits=None):
    hash_func = None
    if params["name"] == "l2lsh":
      hash_func = L2LSH(params["l2lsh"], num_hashes)
    if params["name"] == "l2lsh_torch":
      hash_func = L2LSH_TORCH(params["l2lsh_torch"], num_hashes)
    elif params["name"] == "srp":
      hash_func = SRP(params["srp"], num_hashes)
    elif params["name"] == "srp_torch":
      hash_func = SRP_TORCH(params["srp_torch"], num_hashes, num_bits)
    elif params["name"] == "hist":
      hash_func = HISTOGRAM(params["hist"])
    else:
      raise NotImplementedError
    return hash_func
