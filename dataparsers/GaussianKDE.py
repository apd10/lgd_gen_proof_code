import numpy as np
import torch
from torch.utils import data
import pandas as pd
import pdb
import pickle
import concurrent
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scipy
from BasicData import BasicData

from dataparsers.BasicParser import *


class GaussianKDE(BasicParser):
    def __init__(self, datafile, params):
        super(GaussianKDE, self).__init__()
        self.sigma = params["sigma"]
        self.length = params["epoch_samples"]
        self.kde_sample_ds = BasicData(params["underlying_data"])
        self.data_dimension = self.kde_sample_ds.dim()
        self.data_len = self.kde_sample_ds.len()
        self.gaussian_var =  np.identity(self.data_dimension) * self.sigma**2
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = np.random.randint(0, self.data_len)
        bsample, label = self.kde_sample_ds.getitem(idx) # we need random 
        sample = np.random.multivariate_normal(bsample, self.gaussian_var)
        return sample, label
      
