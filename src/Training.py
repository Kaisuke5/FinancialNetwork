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

    ## Set hyperparameters
    def __init__(self,dataname):
        self.dataname = dataname
        self.Lay = 3
        self.batchsize = 100     # size of batch
        self.n_epoch = 5         # num of back prop. iteration
        self.n_units = 100       # num of units in hidden layer
        self.n_output = 2        # num of units in output layer
        self.dropout = 0         # dropout rate
        self.train_ac,self.test_ac,self.train_mean_loss,self.test_mean_loss = [],[],[],[]


    ## Standardization function
    def stdinp(self):
        self.M = np.mean(self.D['data'],axis=0)
        self.Sd = np.std(self.D['data'],axis=0)
        stmat = np.zeros([len(self.Sd),len(self.Sd)])
        for i in range(0,len(self.Sd)):
            stmat[i][i] = self.Sd[i]
        S_inv = la.inv(np.matrix(stmat))
        input_s = S_inv.dot((np.matrix(self.D['data'] - self.M)).T)
        self.D['data'] = np.array(input_s.T)


    ## Prepare dataset
    def load(self):
        print('load dataset pkl file')
        with open(self.dataname, 'rb') as D_pickle:
            self.D = six.moves.cPickle.load(D_pickle)
        self.D['data'] = np.array(self.D['data'])             # to np array
        self.stdinp()                                         # standardize input
        self.D['data'] = self.D['data'].astype(np.float32)    # 32 bit expression needed for chainer
        self.D['target'] = np.array(self.D['target'])         # to np array
        self.D['target'] = self.D['target'].astype(np.int32)  # 32 bit expression needed for chainer

        self.n_input = len(self.D['data'][0])
        self.N = len(self.D['data'])*8/10       ## split data into two subsets: for training and test
        self.x_train, self.x_test = np.split(self.D['data'],   [self.N])
        self.y_train, self.y_test = np.split(self.D['target'], [self.N])
        self.N_test = self.y_test.size


    ## Prepare multi-layer perceptron model: l1=input nodes, l3=output nodes
    def setmodel(self):
        self.model = chainer.FunctionSet(l1=F.Linear(self.n_input, self.n_units),
                                    l2=F.Linear(self.n_units, self.n_units),
                                    l3=F.Linear(self.n_units, self.n_output))
        ## Setup optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)


    ## Neural net architecture
    ## softmax and accuracy for discrimination task
    def forward(self, x_data, y_data, dropout, train=True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        h1 = F.dropout(F.sigmoid(self.model.l1(x)), ratio=dropout, train=train)
        h2 = F.dropout(F.sigmoid(self.model.l2(h1)), ratio=dropout, train=train)
        y = self.model.l3(h2)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


    def train(self):
        perm = np.random.permutation(self.N)
        sum_accuracy = 0
        sum_loss = 0
        # batch loop
        for i in six.moves.range(0, self.N, self.batchsize):
            x_batch = np.asarray(self.x_train[perm[i:i + self.batchsize]])
            y_batch = np.asarray(self.y_train[perm[i:i + self.batchsize]])
            self.optimizer.zero_grads()
            loss, acc = self.forward(x_batch, y_batch, self.dropout)
            loss.backward()
            self.optimizer.update()
            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)
        self.train_mean_loss.append(sum_loss / self.N)
        self.train_ac.append(sum_accuracy / self.N)


    ## evaluation
    def evaluate(self):
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, self.N_test, self.batchsize):
            x_batch = np.asarray(self.x_test[i:i + self.batchsize])
            y_batch = np.asarray(self.y_test[i:i + self.batchsize])
            loss, acc = self.forward(x_batch, y_batch, self.dropout, train=False)
            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)
        self.test_mean_loss.append(sum_loss / self.N_test)
        self.test_ac.append(sum_accuracy / self.N_test)

    ## LOGGING
    def writelog(self,stime,etime):
        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.info('New trial.\nData: %s\nAll data: %d frames, train: %d frames / test: %d frames.\n   Layers = %d, Units= %d, Batchsize = %d,  Time = %.3f,  Dropout = %.3f\n   Epoch: 0,  test mean loss=  %.5f, accuracy=  %.5f\n   Epoch: %d, train mean loss=  %.5f, accuracy=  %.5f\n              test mean loss=  %.3f, accuracy=  %.3f\n',
                     self.dataname,self.N+self.N_test,self.N,self.N_test,self.Lay,self.n_units,self.batchsize,etime-stime,self.dropout,self.test_mean_loss[0], self.test_ac[0],self.n_epoch, self.train_mean_loss[-1], self.train_ac[-1],self.test_mean_loss[-1], self.test_ac[-1])
        f = open(LOG_FILENAME, 'rt')
        try:
            body = f.read()
        finally:
            f.close()
        print body


if __name__=="__main__":
    dataname = "AMZN5.pkl"
    LOG_FILENAME = 'log_train.txt'
    Data = Training(dataname)

    Data.load()
    Data.setmodel()

    stime = time.clock()
    for epoch in six.moves.range(0, Data.n_epoch + 1):
        print('epoch', epoch)
        start_time = time.clock()
        if epoch > 0:
            Data.train()
            print('train mean loss={}, accuracy={}'.format(Data.train_mean_loss[-1], Data.train_ac[-1]))
        Data.evaluate()
        print('test  mean loss={}, accuracy={}'.format(Data.test_mean_loss[-1], Data.test_ac[-1]))
        end_time = time.clock()
        print "\ttime = %.3f" %(end_time-start_time)

    etime = time.clock()
    Data.writelog(stime,etime)


