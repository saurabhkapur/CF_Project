import numpy as np
from data import read_user
import pickle
import sys
import math

def cal_precision(p,cut):

    # Dense
    if (int(sys.argv[1]) == 1):
        R_true = read_user('Dataset/cf-test-1-users.dat')
    # Sparse both citeulike-a and citeulike-t
    elif (int(sys.argv[1]) == 2):
        R_true = pickle.load(open("R_test.pkl",'rb'))
    #FlickScore
    elif (int(sys.argv[1]) == 3): 
        R_true,l = pickle.load(open("flk_R_test.pkl",'rb'))
    #Last.fm
    elif (int(sys.argv[1]) == 4): 
        R_true = pickle.load(open("Fm_R_test.pkl",'rb'))

    dir_save = 'cdl'+str(p)
    U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    R = U*V.T
    num_u = R.shape[0]
    num_hit = 0
    mAP = 0.0
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
        # _true = set(np.where(R_true[i,:]>0)[1].A1)
        # print l_rec
        s_true = set(np.where(R_true[i,:]>0)[1])
        # #######################
        # print s_true
        
        Ap = 0.0
        count = 0.0
        for r in range(len(s_rec2)):
            if (s_rec2[r] in s_true):
                count += 1
                Ap += (float(count))/(float(r+1))

        if (count > 0 ):          
            # print Ap/count    
            mAP += Ap/count
        ############################
        # cnt_hit = len(s_rec.intersection(s_true))
        # num_hit += cnt_hit
    if (int(sys.argv[1]) == 3):
        error = 0.0
        error_rmse = 0.0
        cnt = 0
        for i,j in l:
            cnt += 1
            error += abs(R_true[i,j] - R[i,j])
            error_rmse += (R_true[i,j] - R[i,j])*(R_true[i,j] - R[i,j])


        print 'MAE: %.3f' % (float(error)/cnt)
        print 'RMSE: %.3f' % (math.sqrt(float(error_rmse)/cnt))

    print np.mean(abs(np.subtract(R_true, R)))
    # print 'Precision: %.3f' % (float(num_hit)/num_u/cut)
    print 'mAP: %.3f' % (float(mAP)/num_u)

cal_precision(4,500)


