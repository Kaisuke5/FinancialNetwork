
import numpy as np
from numpy import linalg as la
import datetime
import matplotlib.pyplot as plt
import logging, time


class DataUtilFunc:

    ## Data standardization (only input side)
    def stdinp(self,input):
        M = np.mean(input,axis=0)
        Sd = np.std(input,axis=0)
        stmat = np.zeros([len(Sd),len(Sd)])
        for i in range(0,len(Sd)):
            stmat[i][i] = Sd[i]
        S_inv = la.inv(np.matrix(stmat))
        input_s = S_inv.dot((np.matrix(input - M)).T)
        input_s = np.array(input_s.T)
        return input_s, M, Sd


    ## Plot func
    def regression_acc_plot(self,target,predict,epoch,compname):
        filename=compname+"_acc_refression"+".jpg"
        if epoch==0:
            plt.plot(target,"b")                         
        elif epoch==epoch/10:
            plt.plot(predict,"r")
        elif epoch==self.n_epoch:
            plt.plot(predict,"g")
            plt.savefig(filename)
            plt.close()

    def acc_plot(self,train,test,compname):
        filename=compname+"_acc"+".jpg"
        plt.plot(train,"b")
        plt.plot(test,"r")
        plt.savefig(filename)
        plt.close()

    def meanloss_plot(self,train,test,compname):
        filename=compname+"_meanloss"+".jpg"
        plt.plot(train,"b")
        plt.plot(test,"r")
        plt.savefig(filename)
        plt.close()

    ## Logging
    def writelog(self,stime,etime,compname,N,N_test,Lay,n_units,n_output,n_epoch,batchsize,dropout,train_ml,test_ml,train_ac,test_ac):
		LOG_FILENAME = '../log/log'+compname+'.txt'
		logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s %(message)s')
		if n_output>1:            
		    logging.info('New trial: Discrimination\nData: %s\nAll data: %d frames, train: %d frames / test: %d frames.\n   Layers = %d, Units= %d, Batchsize = %d,  Time = %.3f,  Dropout = %.3f\n   Epoch: 0,  test mean loss=  %.5f, accuracy=  %.5f\n   Epoch: %d, train mean loss=  %.5f, accuracy=  %.5f\n              test mean loss=  %.3f, accuracy=  %.3f\n',
		                 compname,N+N_test,N,N_test,Lay,n_units,batchsize,etime-stime,dropout,test_ml[0], test_ac[0],n_epoch, train_ml[-1], train_ac[-1],test_ml[-1], test_ac[-1])
		else:
		    logging.info('New trial: Regression\nData: %s\nAll data: %d frames, train: %d frames / test: %d frames.\n   Layers = %d, Units= %d, Batchsize = %d,  Time = %.3f,  Dropout = %.3f\n   Epoch: 0,  test mean loss=  %.5f\n   Epoch: %d, train mean loss=  %.5f\n              test mean loss=  %.3f\n',
		                 compname,N+N_test,N,N_test,Lay,n_units,batchsize,etime-stime,dropout,test_ml[0],n_epoch, train_ml[-1],test_ml[-1])
		f = open(LOG_FILENAME, 'rt')
		try:
		    body = f.read()
		finally:
		    f.close()


	## save model
    def savemodel(self,model,compname):
        now = datetime.datetime.now()
        datestr = now.strftime("_%Y%m%d_%H%M%S")

        savename = compname+datestr+'.pkl'
        print('Save models as pkl...')
        with open(savename, 'wb') as output:
            six.moves.cPickle.dump(model, output, -1)
        print('Done')




"""

    def writelog(self,stime,etime,LOG_FILENAME):
        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s %(message)s')

        if self.n_output>1:            
            logging.info('New trial: Discrimination\nData: %s\nAll data: %d frames, train: %d frames / test: %d frames.\n   Layers = %d, Units= %d, Batchsize = %d,  Time = %.3f,  Dropout = %.3f\n   Epoch: 0,  test mean loss=  %.5f, accuracy=  %.5f\n   Epoch: %d, train mean loss=  %.5f, accuracy=  %.5f\n              test mean loss=  %.3f, accuracy=  %.3f\n',
                         self.dataname,self.N+self.N_test,self.N,self.N_test,self.Lay,self.n_units,self.batchsize,etime-stime,self.dropout,self.test_mean_loss[0], self.test_ac[0],self.n_epoch, self.train_mean_loss[-1], self.train_ac[-1],self.test_mean_loss[-1], self.test_ac[-1])
        else:
            logging.info('New trial: Regression\nData: %s\nAll data: %d frames, train: %d frames / test: %d frames.\n   Layers = %d, Units= %d, Batchsize = %d,  Time = %.3f,  Dropout = %.3f\n   Epoch: 0,  test mean loss=  %.5f\n   Epoch: %d, train mean loss=  %.5f\n              test mean loss=  %.3f\n',
                         self.dataname,self.N+self.N_test,self.N,self.N_test,self.Lay,self.n_units,self.batchsize,etime-stime,self.dropout,self.test_mean_loss[0],self.n_epoch, self.train_mean_loss[-1],self.test_mean_loss[-1])
        f = open(LOG_FILENAME, 'rt')
        try:
            body = f.read()
        finally:
            f.close()
"""
