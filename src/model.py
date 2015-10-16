#coding:utf-8
from chainer import FunctionSet, Variable
import chainer.functions as F
 
# 多層パーセプトロンの定義


model = FunctionSet(l1=F.Linear( 784, 1000),
                    l2=F.Linear(1000, 1000),
                    l3=F.Linear(1000, 10))
def forward(x_data, y_data):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(h1))
    y  = model.l3(h2)
    return F.softmax_cross_entropy(y, t)
 
# 勾配計算
x_data, y_data = ...            # ミニバッチを初期化
loss = forward(x_data, y_data)  # 順伝播
loss.backward()                 # 逆伝播