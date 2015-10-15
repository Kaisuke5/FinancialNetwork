#coding:utf-8
from yahoo_finance import Share
from pprint import pprint
import matplotlib.pyplot as plt
from YahooAggregator import yahoo_aggregator 
import pickle
import six



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
	# x_var_high:最高値の変化量
	# x_var_low:最高値の変化量
	# x_var_vol:取引量

	#これをk日分を一つの入力
	#ある時点からterm日間でその時点の株価からper%分値上がりしてたら1を正解データへ

	def make_train_data(self,term,k,per):

		#何日おきにつくるか
		N=1

		print"making trainning data"
		size=len(self.yahoo_data["aver_price"])
		
		#print len(self.yahoo_data["aver_price"]),len(self.yahoo_data["var_low"]),len(self.yahoo_data["var_high"])

		x_var_high=[]
		x_var_low=[]
		x_var_vol=[]
		y_list=[]

		day=k

		#targetをつくる term内にper%あがったら1
		while (day+term)<=size:
			y=0
			for p in self.yahoo_data["aver_price"][day:day+term]:
				if p>=self.yahoo_data["aver_price"][day]*float(1+per):
					y=1
					break

			x_var_high.append(list(self.yahoo_data["var_high"][day-k:day]))
			x_var_low.append(list(self.yahoo_data["var_low"][day-k:day]))
			x_var_vol.append(list(self.yahoo_data["var_vol"][day-k:day]))
			y_list.append(y)
			
			#何日おきにつくるか
			day+=N

		output={"var_high":x_var_high,"var_low":x_var_low,"var_vol":x_var_vol,"target":y_list}
		print "taget rate %d/%d" % (len(filter(lambda n:n==1,output["target"])),len(output["target"]))
		return output


	
	def make_pikle(self,output):
		filename="../data/"+".pkl"
		Data= {}
		Data['data'] = [0 for raw in output["target"]]
		Data['target'] = [0 for raw in output["target"]]


		for i in range(len(output["target"])):
			data=list(output["var_high"][i]+output["var_low"][i]+output["var_vol"][i])
			Data["data"][i]=data
			Data["target"][i]=output["target"][i]


		with open("../data/"+filename,'wb') as output:
			six.moves.cPickle.dump(Data,output,-1)
		print 'Saving: DONE'




		

	def plot(self,filename="graph"):
		plt.plot(self.yahoo_data["aver_price"])
		#plt.show()
		plt.savefig(filename+".jpg")

if __name__=="__main__":
	ya=yahoo_aggregator()
	company_name="AMZN"
	company=Company(company_name)
	company.get_data(ya)

	#20days one feature 
	output=company.make_train_data(30,10,0.3)
	company.make_pikle(output)
	

	c=0
	for i in output["target"]:
		if i ==1: c+=1

	print c,len(output["target"])




"""
	import pickle
	import six
	GOOG= {}
	GOOG['data'] = [0 for raw in output["target"]]
	GOOG['target'] = [0 for raw in output["target"]]


	for i in range(len(output["target"])):
		data=list(output["var_high"][i]+output["var_low"][i]+output["var_vol"][i])
		GOOG["data"][i]=data
		GOOG["target"][i]=output["target"][i]

	print len(GOOG["target"]),len(GOOG["data"][0])

	with open(company_name+'.pkl','wb') as output:
		six.moves.cPickle.dump(GOOG,output,-1)
	print 'Saving: DONE'

"""





