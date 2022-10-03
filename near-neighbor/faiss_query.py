import numpy as np
import time
import faiss

#change these params
d = 128                       # dimension
nb = 1000000000                   # database size, 1B
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
gpu = False

test_datapath = "../data/sift-1b/queries.npy"
nns_datapath = "../data/sift-1b/neighbors100.npy"
test = np.load(test_datapath).astype(np.float32)
nns = np.load(nns_datapath)
print (type(test[0,0]))

res = faiss.StandardGpuResources()  # use a single GPU
# index_path = "./deep100M_ivf65536_pq48AllGPU.index"
index_path = "sift1b_ivf16384_PQ96.index"

cpu_index = faiss.read_index(index_path)
index_ivf = faiss.extract_index_ivf(cpu_index)
if gpu:
    # gpu_index_ivf = faiss.index_cpu_to_gpu(res, 4, index_ivf)
    co = faiss.GpuMultipleClonerOptions()
    gpu_index_ivf = faiss.index_cpu_to_all_gpus(index_ivf, co)
else:
    gpu_index_ivf = index_ivf

# my_index_ivf.nprobe = 123;
k = 100                       # we want to see 4 nearest neighbors
f1 = open(index_path.split('.')[0] + 'sift-1b-Faiss.txt', "a")

# Nprobe = list(range(1,10,2))+ list(range(10,100,10)) + list(range(100,2000,100)) 
# Nprobe = list(range(2000,3000, 500))
Nprobe =  list(range(1,10,2))+ list(range(10,100,20))
# + list(range(100,1000,200)) + list(range(1000,2000,500)) +  list(range(2000,10000,500)) 


for nprobe in Nprobe:
    if gpu:
        for i in range(gpu_index_ivf.count()):
            faiss.downcast_index(gpu_index_ivf.at(i)).nprobe = nprobe
    else:
        gpu_index_ivf.nprobe = nprobe

    labels = np.empty([test.shape[0], k])
    t1 = time.time()

    # batch
    for j in range(test.shape[0]//32):
        D, labels[j*32:(j+1)*32,] = gpu_index_ivf.search(test[j*32:(j+1)*32,:], k)  # actual search

    #no btach
    # for j in range(test.shape[0]):
    #     D, labels[j,] = gpu_index_ivf.search(test[j:j+1,:], k)  # actual search

    t2 = time.time()
    qt = t2-t1
    # print ("total test time: ", t2-t1)

    rcl1 = 0
    rcl10 = 0
    rcl100 = 0
    for i in range(labels.shape[0]):
        rcl1 = rcl1 + len(np.intersect1d(np.array(labels[i,:1]), np.array(nns[i,:1])))/1.0
        rcl10 = rcl10 + len(np.intersect1d(np.array(labels[i,:10]), np.array(nns[i,:10])))/10.0
        rcl100 = rcl100 + len(np.intersect1d(np.array(labels[i,:100]), np.array(nns[i,:100])))/100.0

    recall1 = rcl1/len(labels)
    recall10 = rcl10/len(labels)
    recall100 = rcl100/len(labels)
    f1.write(str(nprobe)+','+ str(qt)+','+str(recall1) +','+str(recall10)+','+str(recall100) + '\n')
    print (str(nprobe)+','+ str(qt)+','+str(recall1)+','+str(recall10)+','+str(recall100) + '\n')


f1.close()
