import torch
from torch import nn
from torch.autograd import Variable
from DataSet import DataSet
from Sampler import Sampler
import pdb

class Data:
    ''' specifies the 
        - file 
        - dataparser for the file (dataset)
        - sampler
    '''
    def __init__(self, data_params):
        super(Data, self).__init__()
        self.data_file = data_params["file"]
        self.n_dataset = data_params["dataset"]
        self.dataset = DataSet.get(self.data_file, self.n_dataset, data_params[self.n_dataset])
        self.n_sampler = data_params["sampler"]
        self.sampler = Sampler.get(self.dataset, self.n_sampler, data_params[self.n_sampler])
        
        self.num_points = 0
        self.total_num_points = self.sampler.total_size

    def next(self):
        num, X = self.sampler.next()
        self.num_points += num
        return X
        
    def end(self):
        return self.num_points >= self.total_num_points

    def reset(self):
        self.num_points = 0
        self.sampler.reset()

    def len(self):
        return self.total_num_points

    def batch_size(self):
        return self.sampler.batch_size
