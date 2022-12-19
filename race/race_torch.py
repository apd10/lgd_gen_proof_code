import pdb
import torch
from Hash import *

class RACE_torch():
    def __init__(self, params, repetitions, num_hashes, dimension,dtype = np.float32):
        self.dtype = dtype
        self.R = repetitions # number of ACEs (rows) in the array
        self.W = 2 ** num_hashes  # range of each ACE (width of each row) # ????   
        self.K = num_hashes
        self.D = dimension
        self.N = 0
        self.counts = torch.zeros((self.R,self.W))

        self.hashes = []
        for i in range(self.R):
            self.hashes.append(HashFunction.get(params["lsh_function"], num_hashes=self.K)) 
            
    # increase count(weight) for X (batchsize * dimension)
    def add(self, X, alpha):
        self.N += X.shape[0]
        for i in range(self.R):
            hashcode = self.hashes[i].compute(X)
            self.counts[i].scatter_add_(0,hashcode.type(torch.LongTensor),alpha)

    # batchwise
    def query(self, x):
        mean = []
        for i in range(self.R):
            hc = self.hashes[i].compute(x)
            mean.append(torch.gather(self.counts[i],0,hc.type(torch.LongTensor)))
        return sum(mean)/(self.R)
    
    def print_table(self):
        print(self.counts.round(decimals=2))

