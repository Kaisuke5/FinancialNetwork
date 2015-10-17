#coding:utf-8
""" 
Company class file: converting data and make pickle of ['data'] and ['target']

Kai & Shiba

Last Stable:
2015/10/16

Last updated:
2015/10/16
"""
from yahoo_finance import Share
from pprint import pprint
import matplotlib.pyplot as plt
from YahooAggregator import Yahoo_aggregator 
import pickle
import six

import numpy as np
import os


class Company:

	def __init__(self,name):
		self.share=Share(name)
		self.yahoo_data=[]
		self.yahoo_feature=[]
		self.name=name
		


	#csvからaggregatorでデータをとってくる(まだ加工前)
	def get_data(self,aggregator):
		print "reading data"
		self.yahoo_data=aggregator.get_data(self.share,self.name)
		return self.yahoo_data

	# target
	# x_high:最高値の変化量
	# x_low:最高値の変化量
	# x_vol:取引量

	#これをk日分を一つの入力
	#ある時点からterm日間でその時点の株価からper%分値上がりしてたら1を正解データへ

	def make_train_data(self,term,k,per):

		#何日おきにつくるか
		N=1

		print"making trainning data"
		print self.yahoo_data["aver"]
		size=len(self.yahoo_data["aver"])
		col=size-term-k+1



		#初期化 最初の時点

		self.x_yahoo_data=np.zeros((col,3*k))
		self.y_yahoo_data=np.zeros(col)


		#print len(self.yahoo_data["aver"]),len(self.yahoo_data["low"]),len(self.yahoo_data["high"])
		
		day=k
		index=0
		#targetをつくる term内にper%あがったら1
		while (day+term)<=size:

			maxaver=np.max(self.yahoo_data["aver"][day:day+term])

			x_high=list(self.yahoo_data["high"][day-k:day])
			x_low=list(self.yahoo_data["low"][day-k:day])
			x_vol=list(self.yahoo_data["vol"][day-k:day])

			self.x_yahoo_data[index]=x_high+x_low+x_vol
			self.y_yahoo_data[index]=maxaver/self.yahoo_data["aver"][day]-1
			#何日おきにつくるか
			day+=N
			index+=1
			
			
		output={"data":self.x_yahoo_data,"target":self.y_yahoo_data}
		return output


	
	def make_pickle(self,output):
		filename="../data/"+self.name+".pkl"

		with open(filename,'wb') as output:
			six.moves.cPickle.dump(Data,output,-1)
		print 'Saving: DONE'


	def plot(self,filename="graph"):
		plt.plot(self.yahoo_data["aver"])
		#plt.show()
		plt.savefig(filename+".jpg")

if __name__=="__main__":

	ya=Yahoo_aggregator()
	company_name="GOOG"
	company=Company(company_name)
	company.get_data(ya)
	output=company.make_train_data(30,10,0.3)
	company.make_pickle(output)
	company.plot()
	print np.mean(output["target"])



"""
MEMO::

	import pickle
	import six
	GOOG= {}
	GOOG['data'] = [0 for raw in output["target"]]
	GOOG['target'] = [0 for raw in output["target"]]

	for i in range(len(output["target"])):
		data=list(output["high"][i]+output["low"][i]+output["vol"][i])
		GOOG["data"][i]=data
		GOOG["target"][i]=output["target"][i]

	print len(GOOG["target"]),len(GOOG["data"][0])

	with open(company_name+'.pkl','wb') as output:
		six.moves.cPickle.dump(GOOG,output,-1)
	print 'Saving: DONE'

"""

