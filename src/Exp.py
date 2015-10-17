
from Company import Company
from YahooAggregator import Yahoo_aggregator
from Training import Training
from 




class Exp:



	def __init__(self, complist, aglist, K, term, per):
		self.predcomp = complist.keys()[0]
		self.datacomp = complist.keys()
		self.aglist = aglist
		self.K = K
		self.term = term
		self.per = per


	def comploop(self):


	def comptrain(self,company,ratio=1):
	    dataname = "../data/AMZN.pkl"
	    LOG_FILENAME = '../log/log.txt'

	    Data = Training(dataname)
	    Data.load()
	    Data.setmodel()

	    stime = time.clock()
	    Data.learningloop()
	    etime = time.clock()

	    Data.writelog(stime,etime,LOG_FILENAME)



n回 メインの会社
n*ratio回　他の会社
1回？　メインの会社

complistの1tつめをたーげっととして
モデルをトレーニング
使うデータはaglistとk以降の引数から取得


同じモデルを使用してcomplistの中でトレーニングを繰り返す




if __name__ = '__main__':
	complist = {"GOOG":1, "AMZN":0.5}
	aglist = ["Y","T"]
	K, term,per = 30,30,0.1

	cp=Company(name)

	Ex = Exp(complist,aglist,K,term,per)


	Exp01.comptrain(Exp01.predcomp,)


	name="AMZN"
	cp=Company(name)

	#get data of yahoo finance from csv
	ya=Yahoo_aggregator()
	cp.get_data(ya)


	#make train_data
	K,term,per=30,30,0.10
	output=cp.make_train_data(term,K,per)
	cp.make_pickle(output)



