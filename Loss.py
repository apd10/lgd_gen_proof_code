import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class Loss:
  def get(params):
    loss_func = None
    if params["name"] == "BCE":
      loss_func = F.binary_cross_entropy
    elif params["name"] == "CE":
      loss_func = F.cross_entropy
    elif params["name"] == "NLL":
      loss_func = F.nll_loss
    elif params["name"] == "MSE":
      loss_func = F.mse_loss
    else:
      raise NotImplementedError
    return loss_func
