import numpy as np
from mult import read_mult
import pickle

def get_mult(i = 1):
    if (i == 1):
        X = read_mult('Dataset/mult.dat',8000).astype(np.float32)
    elif (i ==2): 
        X = read_mult('mult_flkr.txt',4000).astype(np.float32)
    elif (i == 3):
         X = read_mult('Dataset/citeulike-t/mult.dat',20000).astype(np.float32)
    elif (i ==4): 
        X = read_mult('mult_music.txt',4000).astype(np.float32)

    return X

def get_dummy_mult():
    X = np.random.rand(100,100)
    X[X<0.9] = 0
    return X

def read_user(f_in='Dataset/cf-train-1-users.dat',num_u=5551,num_v=16980):
    fp = open(f_in)
    R = np.mat(np.zeros((num_u,num_v)))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)] = 1
            # break
    return R

def read_user_sparse(f_in='Dataset/cf-train-1-users.dat',num_u=5551,num_v=16980):
    fp = open(f_in)
    R = np.mat(np.zeros((num_u,num_v)))
    R_test = np.mat(np.zeros((num_u,num_v)))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        temp = np.random.randint(len(segs))
        # for seg in segs:
        R[i,int(segs[temp])] = 1
        for seg in segs:
            if (segs[temp] != seg):
                R_test[i, int(seg)] = 1
            # else:
                # print "ok"  
    with open("R_test.pkl",'w') as f:
        pickle.dump(R_test,f)
    return R

def read_dummy_user():
    R = np.mat(np.random.rand(100,100))
    R[R<0.9] = 0
    R[R>0.8] = 1
    return R

