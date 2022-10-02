import numpy as np
import pickle
import sys
import os
import pdb

ignore_1 = False
fname = sys.argv[1]

if len(sys.argv) > 2:
  ignore_1 = int(sys.argv[2])

with open(fname, "rb") as f:
    d = pickle.load(f)

cs = d["params"]["num_classes"]
rs = d["params"]["repetitions"]
l = []
for c in range(cs):
    for r in range(rs):
        X = d["memory"][0][0]['topK']
        if ignore_1:
            X = X[X['value'] > 1]
        l.append(X.values.astype(int))
BS = np.concatenate(l)
BS1 = BS[:,:-1].astype(np.int16) # storing the partition ids
BS2 = BS[:,-1].astype(np.int32) # last column
_byt = (16*BS1.shape[0]*BS1.shape[1] + 32*BS2.shape[0]) / 8
print('{},{},{},{}'.format( _byt/ 10**6, "MB", _byt / 10**3, "KB"))
##W = np.concatenate([np.array(d["hashfunction"]["W"]), np.array(d["hashfunction"]["b"]).reshape(1,-1) ])
##np.savez_compressed("w.npz", W)
#np.savez_compressed("bs.npz", BS1)
#os.system("zip bs.zip bs.npz")
#statinfo = os.stat("bs.zip")
#sz1 = statinfo.st_size
#np.savez_compressed("bs.npz", BS2)
#os.system("zip bs.zip bs.npz")
#statinfo = os.stat("bs.zip")
#sz2 = statinfo.st_size
#sz = (sz1+sz2)
#print('{},{},{},{}'.format(sz // 10**5 / 10, "MB", sz // 10**2 / 10, "KB"))
