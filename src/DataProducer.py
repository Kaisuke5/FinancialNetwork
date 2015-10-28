#coding:utf-8
""" 
Company class file: converting data and make pickle of ['data'] and ['target']

Kai & Shiba

Last Stable:
2015/10/16

Last updated:
2015/10/16
"""
#from yahoo_finance import Share
from pprint import pprint
import matplotlib.pyplot as plt
#from YahooAggregator import Yahoo_aggregator 
import pickle
import six
import csv

import numpy as np
import os
from CsvHandler import CsvHandler


class DataProducer():

	def __init__(self,name,now_time="2015-09-30"):
#		self.share=Share(name)
		self.yahoo_data=[]
		self.yahoo_feature=[]
		self.name=name

		self.now_time = now_time
		self.DIR_NAME="../data/"
		self.ch=CsvHandler(name,nowtime)


#csvをチェック
#CsvHandler.get_datai

	#csvからaggregatorでデータをとってくる(まだ加工前)

	# target
	# x_high:最高値の変化量
	# x_low:最高値の変化量
	# x_vol:取引量

	def make_datemap(self,term,k):
		#何日おきにつくるか
		print"making datemap csv..."
		N=1
		self.yahoo_data=self.get_data(self.name)
		size=len(self.yahoo_data["aver"])
		obj = self.read_csv(self.name)
		g=open(self.DIR_NAME+self.name+"_datemap.csv","w")
		fieldnames=("Column","Start","End","TargetDay")
		headers = dict( (n,n) for n in fieldnames )
		writer=csv.DictWriter(g,fieldnames)
		writer.writerow(headers)

		count = 1
		day=k
		while (day+term)<=size:
			row=""
			start = obj[day-k]["Date"]
			end = obj[day]["Date"]
			targetday = obj[day+term-1]["Date"]
			writer.writerow({"Column":count,"Start":start,"End":end,"TargetDay":targetday})
			count+=1
			day+=N
		g.close()
		print 'Saving datemap: DONE'


	def make_train_data(self,term,k,per):
		#何日おきにつくるか
		N=1
		self.yahoo_data=self.get_data(self.name)
		print"making trainning data..."
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
			x_high=list(self.yahoo_data["high"][day-k:day+1])
			x_low=list(self.yahoo_data["low"][day-k:day+1])
			x_vol=list(self.yahoo_data["vol"][day-k:day+1])

			self.x_yahoo_data[index]=x_high+x_low+x_vol
			self.y_yahoo_data[index]=maxaver/self.yahoo_data["aver"][day]-1
			day+=N
			index+=1

		output={"data":self.x_yahoo_data,"target":self.y_yahoo_data}
		return output


	def make_pickle(self,Data):
		filename="../data/"+self.name+".pkl"
		with open(filename,'wb') as output:
			six.moves.cPickle.dump(Data,output,-1)
		print 'Saving pickle: DONE'


	def plot(self,filename="graph"):
		plt.plot(self.yahoo_data["aver"])
		#plt.show()
		plt.savefig(filename+".jpg")



if __name__=="__main__":
	V = DataProducer("GOOG")
	term, k = 10,10
	per = 0.1

	out = V.make_train_data(term,k,per)
	V.make_pickle(out)

	V.make_datemap(term,k)




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


		#これをk日分を一つの入力
	#ある時点からterm日間でその時点の株価からper%分値上がりしてたら1を正解データへ


	ya=Yahoo_aggregator()
	company_name="AMZN"
	company=Company(company_name)
	company.get_data(ya)
	output=company.make_train_data(20,100,0.3)
	company.make_pickle(output)
	company.plot()


"""

"""
from DataProducer import DataProducer
V = DataProducer("GOOG")
#out = V.make_train_data(10,10,0.1)
V.make_datemap(10,10)
"""





