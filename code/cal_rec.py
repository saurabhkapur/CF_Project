import numpy as np
from data import read_user
import pickle
import sys


def cal_rec(p,cut):
    R_true = read_user('Dataset/cf-test-1-users.dat')
    dir_save = 'cdl'+str(p)
    U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    R = U*V.T
    num_u = R.shape[0]
    num_hit = 0
    fp = open(dir_save+'/rec-list.dat','w')
    for i in range(num_u):
        if i!=0 and i%100==0:
            print 'User '+str(i)
        l_score = R[i,:].A1.tolist()
        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        s_true = set(np.ravel(np.where(R_true[i,:]>0)[1]))
        cnt_hit = len(s_rec.intersection(s_true))
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str,l_rec)))
        fp.write('\n')
    fp.close()

def cal_rec_movies(p,cut):
    R_true,l = pickle.load(open("flk_R_test.pkl",'rb'))    
    movie_name = pickle.load(open("flk_movie_name.pkl",'rb')) 
    dir_save = 'cdl'+str(p)
    U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    R = U*V.T
    num_u = R.shape[0]
    num_hit = 0
    fp = open(dir_save+'/rec-list_movie.dat','w')
    for i in range(num_u):
        if i!=0 and i%100==0:
            print 'User '+str(i)
        l_score = R[i,:].A1.tolist()
        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        s_true = set(np.ravel(np.where(R_true[i,:]>0)[1]))
        cnt_hit = len(s_rec.intersection(s_true))
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str,l_rec)))
        fp.write('\n')
    fp.close()

def cal_rec_music(p,cut):
    R_true = pickle.load(open("Fm_R_test.pkl",'rb'))    
    # artist_name = pickle.load(open("flk_artist_name.pkl",'rb')) 
    dir_save = 'cdl'+str(p)
    U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    R = U*V.T
    num_u = R.shape[0]
    num_hit = 0
    fp = open(dir_save+'/rec-list_movie.dat','w')
    for i in range(num_u):
        if i!=0 and i%100==0:
            print 'User '+str(i)
        l_score = R[i,:].A1.tolist()
        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
        l_rec = list(zip(*pl)[0])[:cut]
        s_rec = set(l_rec)
        s_true = set(np.ravel(np.where(R_true[i,:]>0)[1]))
        cnt_hit = len(s_rec.intersection(s_true))
        fp.write('%d:' % cnt_hit)
        fp.write(' '.join(map(str,l_rec)))
        fp.write('\n')
    fp.close()

# citeulike
if (int(sys.argv[1]) == 1):
    cal_rec(4,8)
# music
if (int(sys.argv[1]) == 2):
    cal_rec_music(4,8)
# movies
if (int(sys.argv[1]) == 3):
    cal_rec_movies(4,10)
