
import numpy as np
import time
import faiss 


np.random.seed(1234)             # make reproducible
datapath = "amazonQA/train_384.npy"  #reaplace this with the data file
xb = np.load(datapath)
d = xb.shape[1]                         # dimension
nb = xb.shape[0]                  # database size, 1M

xb = xb.astype(np.float32)
print (type(xb[0,0]))
res = faiss.StandardGpuResources()  # use a single GPU

## Using an IVF index
t1 = time.time()

# index_ivf = faiss.index_factory(d, 'IVF65536,PQ96', faiss.METRIC_INNER_PRODUCT) # IVF\sqrt(N)
index_ivf = faiss.index_factory(d, 'IVF16384,PQ96', faiss.METRIC_INNER_PRODUCT)

# make it an IVF GPU index
print (res)
# gpu_index_ivf = faiss.index_cpu_to_gpu(res, 5, index_ivf) 

assert not index_ivf.is_trained
print ("traning")
index_ivf.train(xb)        # train with nb vectors
assert index_ivf.is_trained

# print ("adding")
# index_ivf.add(xb)          # add vectors to the index
# print("ntotal after ivf: ",index_ivf.ntotal)

print ("total train time: ", time.time()-t1)
saveLoc = "./amazonQA_ivf16384_PQ96.index"
print ("saving at ", saveLoc)
faiss.write_index(faiss.index_gpu_to_cpu(index_ivf), saveLoc)
t2 = time.time()
print ("total code time: ", t2-t1)
