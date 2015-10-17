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
import logging,time
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers



class Training:
    """ Training """

    #################################### Create instance #####################################
    ## Set hyperparameters
    def __init__(self,dataname):
        self.dataname = dataname
        self.Lay = 3
        self.batchsize = 100     # size of batch
        self.n_epoch = 5         # num of back prop. iteration
        self.n_units = 100       # num of units in hidden layer
        self.n_output = 1        			# num of units in output layer

        self.dropout = 0         # dropout rate
        self.train_ac,self.test_ac,self.train_mean_loss,self.test_mean_loss = [],[],[],[]


    #################################### Create instance #####################################
    ## Prepare multi-layer perceptron model: l1=input nodes, l3=output nodes
    def setmodel(self):
        self.model = chainer.FunctionSet(l1=F.Linear(self.n_input, self.n_units),
                                    l2=F.Linear(self.n_units, self.n_units),
                                    l3=F.Linear(self.n_units, self.n_output))
        ## Setup optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)


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


    #################################### Method #####################################
    ## Prepare dataset
    def load(self):
        print('load dataset pkl file')
        with open(self.dataname, 'rb') as D_pickle:
            D = six.moves.cPickle.load(D_pickle)
        self.data = np.array(D['data'])             # to np array
        self.stdinp()                                         # standardize input
        self.data = self.data.astype(np.float32)    # 32 bit expression needed for chainer

        if self.n_output>1:
            self.target = np.array(D['target'])         # to np array
            self.target = self.target.astype(np.int32)  # 32 bit expression needed for chainer
        else:
            self.target = np.array(D['target'])         # to np array
            self.target = self.target.astype(np.float32)  # 32 bit expression needed for chainer

        self.n_input = len(self.data[0])

        self.N = len(self.data)*8/10       ## split data into two subsets: for training and test
        self.x_train, self.x_test = np.split(self.data,   [self.N])
        self.y_train, self.y_test = np.split(self.target, [self.N])
        self.N_test = self.y_test.size

    #################################### Method #####################################
    ## Neural net architecture
    ## softmax and accuracy for discrimination task
    def forward(self, x_data, y_data, dropout, train=True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        h1 = F.dropout(F.sigmoid(self.model.l1(x)), ratio=dropout, train=train)
        h2 = F.dropout(F.sigmoid(self.model.l2(h1)), ratio=dropout, train=train)
        if self.n_output>1:
            y = self.model.l3(h2)
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            y = F.dropout(self.model.l3(h2), ratio=dropout, train=train)
            return F.mean_squared_error(y,t)

    #################################### Method #####################################
    ## Train
    def train(self):
        perm = np.random.permutation(self.N)
        sum_accuracy = 0
        sum_loss = 0
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
                loss = self.forward(x_batch, y_batch, self.dropout)
                loss.backward()
                self.optimizer.update()
                sum_loss += float(loss.data) * len(y_batch)
        self.train_mean_loss.append(sum_loss / self.N)
        if self.n_output>1: self.train_ac.append(sum_accuracy / self.N)


    #################################### Method #####################################
    ## Test
    def test(self):
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, self.N_test, self.batchsize):
            x_batch = np.asarray(self.x_test[i:i + self.batchsize])
            y_batch = np.asarray(self.y_test[i:i + self.batchsize])
            if self.n_output>1:
                loss, acc = self.forward(x_batch, y_batch, self.dropout, train=False)
                sum_loss += float(loss.data) * len(y_batch)
                sum_accuracy += float(acc.data) * len(y_batch)
            else:
                loss = self.forward(x_batch, y_batch, self.dropout, train=False)
                sum_loss += float(loss.data) * len(y_batch)
        self.test_mean_loss.append(sum_loss / self.N_test)
        if self.n_output>1: self.test_ac.append(sum_accuracy / self.N_test)


    #################################### Method #####################################
    ## Learning loop
    def learningloop(self):
        for epoch in six.moves.range(0, self.n_epoch + 1):
            print('epoch', epoch)
            start_time = time.clock()
            if epoch > 0:
                self.train()
                if self.n_output>1:
                    print('train mean loss={}, accuracy={}'.format(self.train_mean_loss[-1], self.train_ac[-1]))
                else:
                    print('train mean loss={}'.format(self.train_mean_loss[-1]))

            self.test()
            if self.n_output>1:
                print('test  mean loss={}, accuracy={}'.format(self.test_mean_loss[-1], self.test_ac[-1]))
            else:
                print('test  mean loss={}'.format(self.test_mean_loss[-1]))
            end_time = time.clock()
            print "\ttime = %.3f" %(end_time-start_time)


    #################################### Method #####################################
    #def evaluate(self):
    #    origdist = np.mean(self.target)            # proportion of 1 in target
    #    pr = (np.array(self.train_ac)*self.N + np.array(self.test_ac[1:])*self.N_test) / (self.N+self.N_test)



    #################################### Method #####################################
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
        print body


if __name__=="__main__":
    dataname = "../data/AMZN.pkl"
    LOG_FILENAME = '../log/log.txt'

    Data = Training(dataname)
    Data.load()
    Data.setmodel()

    stime = time.clock()
    Data.learningloop()
    etime = time.clock()

    Data.writelog(stime,etime,LOG_FILENAME)


