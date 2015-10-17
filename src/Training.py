#!/usr/bin/env python
"""
Two-value discrimination with Chainer

Kai & Shiba

Last Stable:
2015/10/16

Last updated: s
2015/10/16
"""

import argparse
import numpy as np
from numpy import linalg as la
import six
import logging,time,datetime
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import matplotlib.pyplot as plt


class Training:
    """ Training """

    ## Set hyperparameters
    def __init__(self,dataname,epoch=20,n_output=1,Model=None):
        self.datadir = "../data/"
        self.dataname = dataname
        self.n_epoch = epoch         # num of back prop. iteration
        self.Lay = 3
        self.batchsize = 100     # size of batch
        self.n_units = 100       # num of units in hidden layer
        self.n_output = n_output        			# num of units in output layer

        self.dropout = 0.2         # dropout rate
        self.train_ac,self.test_ac,self.train_mean_loss,self.test_mean_loss = [],[],[],[]
        self.load()
        if Model==None:
            self.setmodel()
        else: self.model = Model


    ## Prepare multi-layer perceptron model: l1=input nodes, l3=output nodes
    def setmodel(self):
        self.model = chainer.FunctionSet(l1=F.Linear(self.n_input, self.n_units),
                                        l2=F.Linear(self.n_units, self.n_units),
                                        l3=F.Linear(self.n_units, self.n_output))
        ## Setup optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    ## Prepare dataset
    def load(self):
        if type(self.dataname)==str:
            print('load dataset pkl file')
            with open(self.datadir+self.dataname+".pkl", 'rb') as D_pickle:
                D = six.moves.cPickle.load(D_pickle)
        else: D = self.dataname.copy
        self.data = np.array(D['data'])             # to np array
        self.stdinp()                                         # standardize input
        self.data = self.data.astype(np.float32)    # 32 bit expression needed for chainer

        if self.n_output>1:
            self.target = np.array(D['target']).astype(np.int32)         # to np array
        else:
            self.target = np.array(D['target'])         # to np array
            self.target = self.target.astype(np.float32).reshape(len(self.target), 1)  # 32 bit expression needed for chainer
        self.n_input = len(self.data[0])

        self.N = len(self.data)*95/100       ## split data into two subsets: for training and test
        self.x_train, self.x_test = np.split(self.data,   [self.N])
        self.y_train, self.y_test = np.split(self.target, [self.N])
        self.N_test = self.y_test.size


    #################################### Method #####################################
    ## Data standardization (only input side)
    def stdinp(self):
        self.M = np.mean(self.data,axis=0)
        self.Sd = np.std(self.data,axis=0)
        stmat = np.zeros([len(self.Sd),len(self.Sd)])
        for i in range(0,len(self.Sd)):
            stmat[i][i] = self.Sd[i]
        S_inv = la.inv(np.matrix(stmat))
        input_s = S_inv.dot((np.matrix(self.data - self.M)).T)
        self.data = np.array(input_s.T)


    ## Neural net architecture
    ## softmax and accuracy for discrimination, mse for regression
    def forward(self, x_data, y_data, dropout, train=True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        h1 = F.dropout(F.sigmoid(self.model.l1(x)), ratio=dropout, train=train)
        h2 = F.dropout(F.sigmoid(self.model.l2(h1)), ratio=dropout, train=train)
        if self.n_output>1:
            y = self.model.l3(h2)
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            y = F.dropout(self.model.l3(h2), ratio=dropout, train=train)
            return F.mean_squared_error(y,t), t.data, y.data, y_data

    def accuracyplot(self,t,a,n):
        filename=self.dataname+"_accuracy"+".jpg"
        if n==0:
            plt.plot(t,"b")                         
        elif n==self.n_epoch/10:
            plt.plot(a,"r")
        elif n==self.n_epoch:
            plt.plot(a,"g")
            plt.savefig(filename)

    def meanloss_plot(self,train,test):
        filename=self.dataname+"_meanloss"+".jpg"
        plt.plot(train,"b")
        plt.plot(test,"r")
        plt.savefig(filename)
        
    ## Train
    def train(self):
        perm = np.random.permutation(self.N)
        sum_accuracy, sum_loss = 0,0
        for i in six.moves.range(0, self.N, self.batchsize):                        # batch loop
            x_batch = np.asarray(self.x_train[perm[i:i + self.batchsize]])
            y_batch = np.asarray(self.y_train[perm[i:i + self.batchsize]])
            self.optimizer.zero_grads()
            if self.n_output>1:
                loss, acc = self.forward(x_batch, y_batch, self.dropout)
                loss.backward()
                self.optimizer.update()
                sum_loss += float(loss.data) * len(y_batch)
                sum_accuracy += float(acc.data) * len(y_batch)
            else:
                loss, self.T, self.Y, self.Y_data= self.forward(x_batch, y_batch, self.dropout)
                loss.backward()
                self.optimizer.update()
                sum_loss += float(loss.data) * len(y_batch)
        self.train_mean_loss.append(sum_loss / self.N)
        if self.n_output>1: self.train_ac.append(sum_accuracy / self.N)

    ## Test
    def test(self):
        sum_accuracy, sum_loss = 0,0
        if self.n_output>1:
            loss, acc = self.forward(self.x_test, self.y_test, self.dropout, train=False)
            sum_loss += float(loss.data) * len(self.y_test)
            sum_accuracy += float(acc.data) * len(self.y_test)
        else:
            loss, self.T, self.Y, self.Y_data= self.forward(self.x_test, self.y_test, self.dropout, train=False)
            sum_loss += float(loss.data) * len(self.y_test)
        self.test_mean_loss.append(sum_loss / self.N_test)
        if self.n_output>1: self.test_ac.append(sum_accuracy / self.N_test)


    ## Learning loop
    def learningloop(self):
        for epoch in six.moves.range(0, self.n_epoch + 1):
            print('epoch', epoch)
            if epoch > 0:
                self.train()
                if self.n_output>1:
                    print('train mean loss={}, accuracy={}'.format(self.train_mean_loss[-1], self.train_ac[-1]))
                else: print('train mean loss={}'.format(self.train_mean_loss[-1]))

            self.test()
            self.accuracyplot(self.T,self.Y,epoch)             
            if self.n_output>1:
                print('test  mean loss={}, accuracy={}'.format(self.test_mean_loss[-1], self.test_ac[-1]))
            else: print('test  mean loss={}'.format(self.test_mean_loss[-1]))
        plt.close()
        self.meanloss_plot(self.train_mean_loss,self.test_mean_loss[1:])


    ## Logging
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


    def savemodel(self):
        now = datetime.datetime.now()
        datestr = now.strftime("_%Y%m%d_%H%M%S")
        confstr = '_%s_L%d_u%d_b%d_f%d_g%d' %(Conf.actfunc, MS.getlayernum(finemodel), n_units,batchsize,len(usefileindex), gpu)

        savename = 'finetuning'+confstr+datestr+'.pkl'
        print('Save models as pkl...')
        with open(savename, 'wb') as output:
            six.moves.cPickle.dump(finemodel, output, -1)
        print('Done')



if __name__=="__main__":

    dataname = "AMZN"
    LOG_FILENAME = '../log/log.txt'

    Data = Training(dataname,epoch=100,n_output=1)

    stime = time.clock()
    Data.learningloop()
    etime = time.clock()

    Data.writelog(stime,etime,LOG_FILENAME)




"""
#        for i in six.moves.range(0, self.N_test, self.batchsize):
#            x_batch = np.asarray(self.x_test[i:i + self.batchsize])
#            y_batch = np.asarray(self.y_test[i:i + self.batchsize])

#                loss, acc = self.forward(x_batch, y_batch, self.dropout, train=False)
            loss, acc = self.forward(self.x_test, self.y_test, self.dropout, train=False)
#                sum_loss += float(loss.data) * len(y_batch)
#                sum_accuracy += float(acc.data) * len(y_batch)
            sum_loss += float(loss.data) * len(self.y_test)
            sum_accuracy += float(acc.data) * len(self.y_test)
        else:
#                loss, self.T, self.Y, self.Y_data= self.forward_disp(x_batch, y_batch, self.dropout, train=False)
            loss, self.T, self.Y, self.Y_data= self.forward_disp(self.x_test, self.y_test, self.dropout, train=False)
#                sum_loss += float(loss.data) * len(y_batch)

    #################################### Method #####################################
    #def evaluate(self):
    #    origdist = np.mean(self.target)            # proportion of 1 in target
    #    pr = (np.array(self.train_ac)*self.N + np.array(self.test_ac[1:])*self.N_test) / (self.N+self.N_test)


            start_time = time.clock()

            end_time = time.clock()
            print "\ttime = %.3f" %(end_time-start_time)


#        print body

"""


