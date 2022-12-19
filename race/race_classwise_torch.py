import pdb
import torch
from Hash import *

class Race_Classwise_torch():
    def __init__(self, params, dtype = np.float32):
        self.dtype = dtype

        hashfunc_name = params['lsh_function']['name']
        hashfunc_params = params["lsh_function"]
        
        self.R = hashfunc_params[hashfunc_name]['num_hash'] # number of ACEs (rows) in the array
        self.K = hashfunc_params[hashfunc_name]['num_hash']
        self.num_bits = hashfunc_params[hashfunc_name]['num_bits']   
        self.D = hashfunc_params[hashfunc_name]['dimension']
        self.N = 0
        self.num_classes = params["num_classes"]
        self.W = params['num_bucket']  # range of each ACE (width of each row)
        self.counts = torch.zeros((self.R,self.W))
        self.sketch_weighted = {}
        self.sketch_unweighted = {}
                
        
        self.hashfunc = HashFunction.get(hashfunc_params, num_hashes=self.K, num_bits=self.num_bits) # B x num_hash
            
        for i in range(self.num_classes):
            self.sketch_weighted[i] = torch.zeros((self.R,self.W))
            self.sketch_unweighted[i] = torch.zeros((self.R,self.W))
            
    
    # increase count(weight) for X (batchsize * dimension)
    # class independent
    def add(self, x, alpha):
        self.N += x.shape[0]
        hash_vals = self.hashfunc.hash(x) # compute hash values
        hashcode = torch.fmod(hash_vals, self.W) # rehash to map within the number of buckets
        for i in range(hashcode.shape[1]): # hashcode.shape[1] = self.R (num hashes)
            self.counts[i].scatter_add_(0,hashcode[:,i].type(torch.LongTensor),alpha)

    # Batchwise, Class-based
    def add_class(self, x, y, val): # class_based
        class_names = y
        self.N += x.shape[0]
        hash_vals = self.hashfunc.hash(x) # compute hash values
        hashcode = torch.fmod(hash_vals, self.W) # rehash to map within the number of buckets
        for i in range(hashcode.shape[1]): # hashcode.shape[1] = self.R (num hashes)
            for c in np.arange(self.num_classes):
                examples_perclass = hashcode[:,i][class_names == c]
                if examples_perclass.shape[0] > 0:
#                     val = val[class_names == c].reshape(examples_perclass.shape[0],1).to(device=examples_perclass.device) 
                    val_weight = val[class_names == c]

                    self.sketch_weighted[c][i].scatter_add_(0,examples_perclass.type(torch.LongTensor),val_weight)
                    self.sketch_unweighted[c][i].scatter_add_(0,examples_perclass.type(torch.LongTensor),torch.ones_like(val_weight))
 

    # batchwise
    # class independent
    def query(self, x):
        mean = []
        hash_vals = self.hashfunc.hash(x) # compute hash values
        hashcode = torch.fmod(hash_vals, self.W) # rehash and map to the range of number of buckets 
        for i in range(hashcode.shape[1]):# hashcode.shape[1] = self.R (num hashes)
            mean.append(torch.gather(self.counts[i],0,hashcode[:,i].type(torch.LongTensor)))
        return sum(mean)/(self.R)
    

    # Batchwise, Class-based
    def query_class(self, x):
        self.values_weight = torch.zeros((x.shape[0], self.num_classes))
        self.values_unweight = torch.zeros((x.shape[0], self.num_classes))
        hash_vals = self.hashfunc.hash(x) # compute hash values
        hashcode = torch.fmod(hash_vals, self.W) # rehash and map to the range of number of buckets 
        
        for c in range(self.num_classes):
            cts_w = []
            cts_uw = []
            for i in range(hashcode.shape[1]): # hashcode.shape[1] = num hashes or repetiontions
                cts_w.append(torch.gather(self.sketch_weighted[c][i],0,hashcode[:,i].type(torch.LongTensor)))
                cts_uw.append(torch.gather(self.sketch_unweighted[c][i],0,hashcode[:,i].type(torch.LongTensor)))
    
            self.values_weight[:, c] = sum(cts_w)/self.R
            self.values_unweight[:, c] = sum(cts_uw)/self.R
            
        return self.values_weight,self.values_unweight
    
    
    def print_table(self):
        print(self.counts.round(decimals=2))
