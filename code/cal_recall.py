import numpy as np
from data import read_user
import pickle
import sys
import math

def cal_recall(p,cut):

    if (int(sys.argv[1]) == 1):
        R_true = read_user('Dataset/cf-test-1-users.dat')
    elif (int(sys.argv[1]) == 2):
        R_true = pickle.load(open("R_test.pkl",'rb'))
    elif (int(sys.argv[1]) == 3): 
        R_true,l = pickle.load(open("flk_R_test.pkl",'rb'))
    elif (int(sys.argv[1]) == 4): 
        R_true = pickle.load(open("Fm_R_test.pkl",'rb'))

    dir_save = 'cdl'+str(p)
    U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    R = U*V.T
    num_u = R.shape[0]
    num_hit = 0
    mAP = 0.0
    recall = 0.0
    for i in range(num_u):
        # if i!=0 and i%100==0:
            # print 'Iter '+str(i)+':'
        l_score = R[i,:].A1.tolist()
        # print l_score
        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        # print pl[0]
        s_rec = set(l_rec)
        s_rec2 = list(l_rec)
        _true = set(np.ravel(np.where(R_true[i,:]>0)[1]))
        # print l_rec
        s_true = set(np.where(R_true[i,:]>0)[1])

        cnt_hit = len(s_rec.intersection(s_true))
        if (len(s_true)!=0):
            recall += float(cnt_hit)/len(s_true)
        # num_hit += cnt_hit
    print 'recall@%d: %.3f' % (cut,(recall/num_u))

cal_recall(4,100)


