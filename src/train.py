#!/usr/bin/env python
"""
Two-value discrimination with Chainer

Kai & Shiba

Last Stable:
2015/10/15

Last updated: s
2015/10/15
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


company_name="AMZN"
dataname = "../data/"+company_name+".pkl"
LOG_FILENAME = 'log_train.txt'

## Standardization function
def stdinp(input):
    M = np.mean(input,axis=0)
    Sd = np.std(input,axis=0)
    stmat = np.zeros([len(Sd),len(Sd)])
    for i in range(0,len(Sd)):
        stmat[i][i] = Sd[i]
    S_inv = la.inv(np.matrix(stmat))
    input_s = S_inv.dot((np.matrix(input - M)).T)
    input_s = np.array(input_s.T)
    return input_s


## Prepare dataset
print('load dataset pkl file')
with open(dataname, 'rb') as D_pickle:
    D = six.moves.cPickle.load(D_pickle)
D['data'] = np.array(D['data'])             # to np array
D['data'] = stdinp(D['data'])               # standardize input
D['data'] = D['data'].astype(np.float32)    # 32 bit expression needed for chainer
D['target'] = np.array(D['target'])         # to np array
D['target'] = D['target'].astype(np.int32)  # 32 bit expression needed for chainer


## Set hyperparameters
Nmax = len(D['data'])
n_input = len(D['data'][0])
Lay = 3
batchsize = 100     # size of batch
n_epoch = 5         # num of back prop. iteration
n_units = 100       # num of units in hidden layer
n_output = 2        # num of units in output layer
dropout = 0         # dropout rate

## split data into two subsets: for training and test
N = Nmax*8/10
x_train, x_test = np.split(D['data'],   [N])
y_train, y_test = np.split(D['target'], [N])
N_test = y_test.size


## Prepare multi-layer perceptron model: l1=input nodes, l3=output nodes
model = chainer.FunctionSet(l1=F.Linear(n_input, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, n_output))


## Neural net architecture
## softmax and accuracy for discrimination task
def forward(x_data, y_data, dropout=dropout, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.sigmoid(model.l1(x)), ratio=dropout, train=train)
    h2 = F.dropout(F.sigmoid(model.l2(h1)), ratio=dropout, train=train)
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


## Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)


## Learning loop
train_ac,test_ac,train_mean_loss,test_mean_loss = [],[],[],[]
stime = time.clock()
for epoch in six.moves.range(0, n_epoch + 1):
    print('epoch', epoch)
    start_time = time.clock()

    # training
    if epoch > 0:
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        # batch loop
        for i in six.moves.range(0, N, batchsize):
            x_batch = np.asarray(x_train[perm[i:i + batchsize]])
            y_batch = np.asarray(y_train[perm[i:i + batchsize]])
            optimizer.zero_grads()
            loss, acc = forward(x_batch, y_batch, dropout=dropout)
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)
        print('train mean loss={}, accuracy={}'.format(
            sum_loss / N, sum_accuracy / N))
        train_mean_loss.append(sum_loss / N)
        train_ac.append(sum_accuracy / N)

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = np.asarray(x_test[i:i + batchsize])
        y_batch = np.asarray(y_test[i:i + batchsize])
        loss, acc = forward(x_batch, y_batch, dropout=dropout, train=False)
        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)
    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
    test_mean_loss.append(sum_loss / N_test)
    test_ac.append(sum_accuracy / N_test)
    end_time = time.clock()
    print "\ttime = %.3f" %(end_time-start_time)



## LOGGING
etime = time.clock()
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s'
                    )
logging.info('New trial.\nAll data: %d frames, train: %d frames / test: %d frames.\n   Layers = %d, Units= %d, Batchsize = %d,  Time = %.3f,  Dropout = %.3f\n   Epoch: 0,  test mean loss=  %.5f, accuracy=  %.5f\n   Epoch: %d, train mean loss=  %.5f, accuracy=  %.5f\n              test mean loss=  %.3f, accuracy=  %.3f\n',
             N+N_test,N,N_test,Lay,n_units,batchsize,etime-stime,dropout,test_mean_loss[0], test_ac[0],epoch, train_mean_loss[-1], train_ac[-1],test_mean_loss[-1], test_ac[-1])
f = open(LOG_FILENAME, 'rt')
try:
    body = f.read()
finally:
    f.close()
print 'FILE:'
print body






