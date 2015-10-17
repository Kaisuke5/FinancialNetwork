from Company import Company
from YahooAggregator import Yahoo_aggregator
from Training import Training


class Exp(Training):

	def __init__(self, complist, prodlist, K, term, per):
		self.DIR_NAME = "../data/"
		self.predcomp = complist.keys()[0]
		self.datacomp = complist.keys()
		self.prodlist = prodlist
		self.K = K
		self.term = term
		self.per = per

	def comptrain(self,company,ratio=1):
	    dataname = self.DIR_NAME+company+".pkl"
	    LOG_FILENAME = '../log/log.txt'

		cp=Company(company)
		#get data of yahoo finance from csv
		Yahoo = Yahoo_aggregator()
		cp.get_data(Yahoo)
		#cp.get_data(Twitter)

		output=cp.make_train_data(self.prodlist,self.term,self.K,self.per)
		cp.make_pickle(output)

		data = cp.make_train_data()

		if hasattr(self,"model"):
		    Tr = Training(data,epoch=20,n_output=2,Model=self.model)
		else:
		    Tr = Training(data,epoch=20,n_output=2,Model=None)
	    stime = time.clock()
	    Tr.learningloop()
	    etime = time.clock()
	    Tr.writelog(stime,etime,LOG_FILENAME)
	    self.model = Tr.model


n回 メインの会社
n*ratio回　他の会社
1回？　メインの会社

complistの1tつめをたーげっととして
モデルをトレーニング
使うデータはaglistとk以降の引数から取得


同じモデルを使用してcomplistの中でトレーニングを繰り返す




if __name__ = '__main__':
	complist = {"GOOG":1, "AMZN":0.5}
	prodlist = ["Y","T"]
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



