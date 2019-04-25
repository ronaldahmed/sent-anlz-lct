import sys
import numpy as np

import pdb

np.random.seed(42)

data_fn = sys.argv[1]
labels_fn = sys.argv[2]
size = int(sys.argv[3])

data = open(data_fn,'r').read().strip('\n').split('\n')
labels = open(labels_fn,'r').read().strip('\n').split('\n')
labels = np.array([int(x) for x in labels])

pos_idx = np.where(labels==1)[0]
neg_idx = np.where(labels==0)[0]

np.random.shuffle(pos_idx)
np.random.shuffle(neg_idx)


datafile = open(data_fn + ".subs."+str(size//1000)+"k", 'w')
labelfile = open(labels_fn + ".subs."+str(size//1000)+"k", 'w')

idxs = pos_idx[:size//2].tolist() + neg_idx[:size//2].tolist()
np.random.shuffle(idxs)


for idx in idxs:
	print(data[idx],file=datafile)
	print(labels[idx],file=labelfile)

