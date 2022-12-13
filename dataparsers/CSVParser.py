import numpy as np
import torch
from torch.utils import data
import pandas as pd
import pdb
from dataparsers.BasicParser import *

class CSVParser(BasicParser):
    def __init__(self, X_file, params):
        super(CSVParser, self).__init__()
        self.X_file = X_file
        self.sep = params["sep"]
        self.header = None
        if "header" in params:
            self.header = params["header"]
        self.skiprows = None
        if "skiprows" in params:
            self.skiprows = params["skiprows"]
        self.X = pd.read_csv(self.X_file, sep=self.sep, header=self.header, skiprows=self.skiprows)
        print("initial lines", len(self.X))
        self.X = self.X.dropna()
        print("Drop na lines", len(self.X))
        self.X.index = np.arange(len(self.X))
        self.regression= False
        if "regression" in params:
            self.regression = params["regression"]

        self.label_header = None
        if "label_header" in params:
            self.label_header = params["label_header"]
            self.labels =  self.X[self.label_header]
            del self.X[self.label_header]
        else:
            self.labels = np.zeros(self.X.shape[0])

        if "ignore_label" in params:
            self.labels = np.zeros(self.X.shape[0])

        if "normalizer_const" in params:
            self.X = self.X / params["normalizer_const"]
        self.center = False
        if "centering_info" in params:
            self.center = True
            f = params["centering_info"]
            r = np.load(f)
            self.mu = r["mu"]
            self.std = r["std"]

        self.use_only = None
        if "use_only" in params:
            self.use_only = int(params["use_only"])

        self.length = self.X.shape[0]
        self.dimension = self.X.shape[1]
       # self.train_flg = False
#        pdb.set_trace()
       # if 'train' in X_file:
       #     self.train_flg = True
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):
       # pdb.set_trace()
       # label = self.labels[index]
       # data_point = np.array(self.X.loc[index].values)
        label = self.labels.values[index]
        data_point = np.array(self.X.values[index])
        #pdb.set_trace()
        #if self.center:
        #    data_point = (data_point - self.mu) / (self.std + 1e-5)
        if self.use_only is not None:
           # pdb.set_trace()
            #if self.train_flg: # batchwise parsing
            data_point  = data_point[:,:self.use_only]
           # else:
           #     data_point  = data_point[:self.use_only]
        if self.center:
            data_point = (data_point - self.mu) / (self.std + 1e-5)
       # if label > 0:
        return data_point, label

