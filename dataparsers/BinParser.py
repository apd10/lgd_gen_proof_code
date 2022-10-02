import numpy as np
import torch
from torch.utils import data
import pdb
from dataparsers.BasicParser import *

class BinParser(BasicParser):
    def __init__(self, X_file, params):
        super(BinParser, self).__init__()
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

        self.neg_class = False
        if "neg_class" in params:
            self.neg_class = True


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].strip().split(" ")

        label = int(data[0]) - self.class_base_idx
        if self.neg_class and label == -1:
            label = 0
            
        xdata = data[1:]

        for xd in xdata:
            temp = xd
            data_point[int(temp) - self.base_idx] = 1
        return data_point, label

