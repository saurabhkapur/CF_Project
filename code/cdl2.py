import mxnet as mx
import numpy as np
import logging
import data
from math import sqrt
from autoencoder import AutoEncoderModel
import os
import pickle
import sys



lambda_u = 1 # lambda_u in CDL
lambda_v = 10 # lambda_v in CDL
K = 50
p = 4
is_dummy = False
num_iter = 100 # about 68 iterations/epoch, the recommendation results at the end need 100 epochs
batch_size = 256

np.random.seed(1234) # set seed
lv = 1e-2 # lambda_v/lambda_n in CDL
dir_save = 'cdl%d' % p

if not os.path.isdir(dir_save):
    os.system('mkdir %s' % dir_save)
fp = open(dir_save+'/cdl.log','w')
print 'p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d' % (p,lambda_v,lambda_u,lv,K)
fp.write('p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d\n' % \
        (p,lambda_v,lambda_u,lv,K))
fp.close()


if is_dummy:
    X = data.get_dummy_mult()
    R = data.read_dummy_user()
else:

    # Dense
	if (int(sys.argv[1]) == 1):
		X = data.get_mult()
		R = data.read_user()
	# Sparse citeulike-a
	elif (int(sys.argv[1]) == 2):
		X = data.get_mult()
		R = data.read_user_sparse("Dataset/users.dat")
	# Flickscore
	elif(int(sys.argv[1]) == 3):
		R = pickle.load(open("flk_R_train.pkl",'rb'))
		X = data.get_mult(2)
	# Sparse citeulike-t
	elif (int(sys.argv[1]) == 4):
		X = data.get_mult(3)
		R = data.read_user_sparse("Dataset/citeulike-t/users.dat",7947,25975)
	# Last.fm
	elif(int(sys.argv[1]) == 5):
		R = pickle.load(open("Fm_R_train.pkl",'rb'))
		X = data.get_mult(4)




logging.basicConfig(level=logging.INFO)
cdl_model = AutoEncoderModel(mx.cpu(2), [X.shape[1],100,K],
    pt_dropout=0.2, internal_act='relu', output_act='relu')
'''
We use the following code to define the pSDAE stucture mentioned before. fe_loss is the regression loss for the bottleneck layer,
and fr_loss is the reconstruction loss in the last layer.

            fe_loss = mx.symbol.LinearRegressionOutput(data=self.lambda_v_rt*self.encoder,
                label=self.lambda_v_rt*self.V)
            fr_loss = mx.symbol.LinearRegressionOutput(data=self.decoder, label=self.data)
            self.loss = mx.symbol.Group([fe_loss, fr_loss])            
'''



train_X = X
V = np.random.rand(train_X.shape[0],K)/10
lambda_v_rt = np.ones((train_X.shape[0],K))*sqrt(lv)


U, V, theta, BCD_loss = cdl_model.finetune(train_X, R, V, lambda_v_rt, lambda_u,
        lambda_v, dir_save, batch_size,
        num_iter, 'sgd', l_rate=0.1, decay=0.0,
        lr_scheduler=mx.misc.FactorScheduler(20000,0.1))



print 'Training ends.'

cdl_model.save(dir_save+'/cdl_pt.arg')
np.savetxt(dir_save+'/final-U.dat',U,fmt='%.5f',comments='')
np.savetxt(dir_save+'/final-V.dat',V,fmt='%.5f',comments='')
np.savetxt(dir_save+'/final-theta.dat',theta,fmt='%.5f',comments='')

Recon_loss = lambda_v/lv*cdl_model.eval(train_X,V,lambda_v_rt)
print "Training error: %.3f" % (BCD_loss+Recon_loss)
fp = open(dir_save+'/cdl.log','a')
fp.write("Training error: %.3f\n" % (BCD_loss+Recon_loss))
fp.close()







