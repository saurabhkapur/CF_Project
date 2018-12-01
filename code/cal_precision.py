import numpy as np
from data import read_user
def cal_precision(p,cut):
    R_true = read_user('cf-test-1-users.dat')
    dir_save = 'cdl'+str(p)
    U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    R = U*V.T
    num_u = R.shape[0]
    num_hit = 0
    fp = open(dir_save+'/rec-list.dat','w')
    mAP = 0.0
    for i in range(num_u):
        if i!=0 and i%100==0:
            print 'Iter '+str(i)+':'+str(float(num_hit)/i/cut)
        l_score = R[i,:].A1.tolist()
        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        s_rec2 = list(l_rec)
        #_true = set(np.where(R_true[i,:]>0)[1].A1)
        s_true = set(np.where(R_true[i,:]>0)[1])
        # #######################
        # print s_true
        # print s_rec
        Ap = 0.0
        count = 0
        for r in range(len(s_rec2)):
            if (s_rec2[r] in s_true):
                count += 1
                Ap += (1.0*count)/(r+1)
        if (count != 0):              
            mAP += Ap/count
        ############################
        cnt_hit = len(s_rec.intersection(s_true))
        num_hit += cnt_hit
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str,l_rec)))
        fp.write('\n')
    fp.close()
    print 'Precision: %.3f' % (float(num_hit)/num_u/cut)
    print 'mAP: %.3f' % (float(mAP)/num_u)

cal_precision(4,8)

