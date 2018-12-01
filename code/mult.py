import numpy as np

def read_mult(f_in='mult.dat',D=8000):
    fp = open(f_in)
    lines = fp.readlines()
    X = np.zeros((len(lines),D+1))
    for i,line in enumerate(lines):
        strs = line.strip().split(' ')[1:]
        cnt = 0
        for strr in strs:
            cnt += 1
            segs = strr.split(':')
            X[i,int(segs[0])] = float(segs[1])
        if (cnt == 0):
            X[i, D] = 0.001
    arr_max = np.amax(X,axis=1)
    X = (X.T/arr_max).T
    return X
