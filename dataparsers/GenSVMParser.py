import numpy as np
import torch
from torch.utils import data
import pdb

from dataparsers.BasicParser import *
 

class GenSVMFormatParser(BasicParser):
    def __init__(self, X_file, params):
        super(GenSVMFormatParser, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        self.length = len(self.X)
        self.dim = params["dimension"]
        self.base_idx = 0
        if "base_idx" in params:
            self.base_idx = params["base_idx"]

        self.class_base_idx = 0
        if "class_base_idx" in params:
            self.class_base_idx = params["class_base_idx"]
        self.normalizer_const = 1
        if "normalizer_const" in params:
            self.normalizer_const = params["normalizer_const"]

        self.neg_class = False
        if "neg_class" in params:
            self.neg_class = True
        self.center = False
        self.regression = False
        if "regression" in params:
            self.regression = params["regression"]
    
        if "centering_info" in params:
            self.center = True
            f = params["centering_info"]
            r = np.load(f)
            self.mu = r["mu"]
            self.std = r["std"]
            self.no_scale = False
            if "no_scale" in params:
                self.no_scale = True
            if self.regression:
                self.y_mu, self.y_std = r["y_mean"], r["y_std"]


    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].strip().split(" ")

        if not self.regression:
            label = int(data[0]) - self.class_base_idx
            if self.neg_class and label == -1:
                label = 0
        else:
            label = float(data[0])
            
        xdata = data[1:]

        for xd in xdata:
            temp = xd.split(":")
            data_point[int(temp[0]) - self.base_idx] = float(temp[1]) / self.normalizer_const

        if self.center:
            if self.no_scale:
                data_point = (data_point - self.mu)
                if self.regression:
                    label = label - self.y_mu
            else:
                data_point = (data_point - self.mu) / (self.std + 1e-5)
                if self.regression:
                    label = (label - self.y_mu) / (self.y_std + 1e-5)
        return data_point, label

