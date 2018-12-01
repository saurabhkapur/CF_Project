import csv
from data import read_user
import numpy as np
import pickle

p = 4
user_id = 10
# read predicted results
dir_save = 'cdl%d' % p
# csvReader = csv.reader(open('raw-data.csv','rb'))
d_id_title = pickle.load(open("flk_movie_name.pkl",'rb')) 
# for i,row in enumerate(csvReader):
#     if i==0:
#         continue
#     d_id_title[i-1] = row[3]
R_test, l = pickle.load(open("flk_R_test.pkl",'rb'))
R_train = pickle.load(open("flk_R_train.pkl",'rb'))
fp = open(dir_save+'/rec-list_movie.dat')
lines = fp.readlines()

s_test = set(np.ravel(np.where(R_test[user_id,:]>0)[1]))
l_train = np.ravel(np.where(R_train[user_id,:]>0)[1]).tolist()
l_pred = map(int,lines[user_id].strip().split(':')[1].split(' '))
print '#####  Articles in the Training Sets  #####'
for i in l_train:
    print d_id_title[i]
print '\n#####  Articles Recommended (Correct Ones Marked by Stars)  #####'
for i in l_pred:
    if i in s_test:
        print '* '+d_id_title[i]
    else:
        print d_id_title[i]
fp.close()
