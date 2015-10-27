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
import six
import time
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from DataUtilFunc import DataUtilFunc
import sys

class Training(DataUtilFunc):
    """ Training """

    ## Set hyperparameters
    def __init__(self,compname,epoch=20,n_output=1,Model=None):
        self.datadir = "../data/"
        self.compname = compname
        self.n_epoch = epoch         # num of back prop. iteration
        self.Lay = 3
        self.batchsize = 100     # size of batch
        self.n_units = 100       # num of units in hidden layer
        self.n_output = n_output        			# num of units in output layer

        self.dropout = 0.2         # dropout rate
        self.train_ac,self.test_ac,self.train_mean_loss,self.test_mean_loss = [],[],[],[]
        self.load()

        # model setting
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

    	## data loading
        if type(self.compname)==str:
            print('load dataset pkl file')
            with open(self.datadir+self.compname+".pkl", 'rb') as D_pickle:
                D = six.moves.cPickle.load(D_pickle)
        else: D = self.compname.copy
        self.data = np.array(D['data'])             						# to np array
        self.data, self.M, self.Sd = self.stdinp(self.data)                 # standardize input
        self.data = self.data.astype(np.float32)    						# 32 bit expression needed for chainer

        # discrimination or regression
        if self.n_output>1:														
            self.target = np.array(D['target']).astype(np.int32)         	# discrimination task
        else:
            self.target = np.array(D['target'])         					# regression task
            self.target = self.target.astype(np.float32).reshape(len(self.target), 1)  # 32 bit expression needed for chainer
        self.n_input = len(self.data[0])

		## split data into two subsets: for training and test
        self.N = len(self.data)*95/100       
        self.x_train, self.x_test = np.split(self.data,   [self.N])
        self.y_train, self.y_test = np.split(self.target, [self.N])
        self.N_test = self.y_test.size
        

    #################################### Method #####################################
    ## Neural net architecture
    def forward(self, x_data, y_data, dropout, train=True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    
        
        h1 = F.dropout(F.sigmoid(self.model.l1(x)), ratio=dropout, train=train)
        h2 = F.dropout(F.sigmoid(self.model.l2(h1)), ratio=dropout, train=train)

	    ## softmax and accuracy for discrimination, mse for regression
        if self.n_output>1:
            y = self.model.l3(h2)
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
        else:
            y = F.dropout(self.model.l3(h2), ratio=dropout, train=train)
            return F.mean_squared_error(y,t), t.data, y.data, y_data

        
    ## Train
    def train(self):
        perm = np.random.permutation(self.N)
        sum_accuracy, sum_loss = 0,0

		# batch loop        
        for i in six.moves.range(0, self.N, self.batchsize): 
            x_batch = np.asarray(self.x_train[perm[i:i + self.batchsize]])
            y_batch = np.asarray(self.y_train[perm[i:i + self.batchsize]])
            self.optimizer.zero_grads()

            # discrimination or regression
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
        if self.n_output>1: self.train_ac.append(sum_accuracy / self.N)		# only discrimination


    ## Test
    def test(self):
        sum_accuracy, sum_loss = 0,0

        # discrimination or regression
        if self.n_output>1:
            loss, acc = self.forward(self.x_test, self.y_test, self.dropout, train=False)
            sum_loss += float(loss.data) * len(self.y_test)
            sum_accuracy += float(acc.data) * len(self.y_test)
        else:
            loss, self.T, self.Y, self.Y_data= self.forward(self.x_test, self.y_test, self.dropout, train=False)
            sum_loss += float(loss.data) * len(self.y_test)

        self.test_mean_loss.append(sum_loss / self.N_test)
        if self.n_output>1: self.test_ac.append(sum_accuracy / self.N_test)		# only discrimination


    ## Learning loop, including training and test
    def learningloop(self):
        for epoch in six.moves.range(0, self.n_epoch + 1):
            print('epoch', epoch)

            # training
            if epoch > 0:
                self.train()

                # discrimination or regression
                if self.n_output>1:
                    print('train mean loss={}, accuracy={}'.format(self.train_mean_loss[-1], self.train_ac[-1]))
                else:
                	print('train mean loss={}'.format(self.train_mean_loss[-1]))

            # test
            self.test()
            
            # discrimination or regression
            if self.n_output>1:
                print('test  mean loss={}, accuracy={}'.format(self.test_mean_loss[-1], self.test_ac[-1]))
                self.acc_plot(self.train_ac,self.test_ac[1:],self.compname)
            else:
            	print('test  mean loss={}'.format(self.test_mean_loss[-1]))
            	self.regression_acc_plot(self.T,self.Y,epoch,self.compname)

        self.meanloss_plot(self.train_mean_loss,self.test_mean_loss[1:],self.compname)
        if self.n_output==1: self.train_ac, self.test_ac = 0,0



if __name__=="__main__":

    compname = "AAPL"

    Data = Training(compname,epoch=50,n_output=1)

    stime = time.clock()
    Data.learningloop()
    etime = time.clock()


    Data.writelog(stime,etime,compname,Data.N,Data.N_test,Data.Lay,Data.n_units,Data.n_output,
    	Data.n_epoch,Data.batchsize,Data.dropout,Data.train_mean_loss,Data.test_mean_loss,Data.train_ac,Data.test_ac)







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
    Data.writelog(stime,etime,LOG_FILENAME)

"""


