#coding:utf-8
""" 
Read from Yahoo-finance and make csv file

Kai & Shiba

Last Stable:
2015/10/16

Last updated:
2015/10/16
"""
import csv
from pprint import pprint
import numpy as np
import os
from DataProducer import DataProducer


class Yahoo_aggregator(DataProducer):
	def __init__(self):
		pass




if __name__=="__main__":

	ya=Yahoo_aggregator()
	#ya.make_csv("AAPL","2015-09-30")
	ya.make_csv("FB","2015-09-30")
	#ya.make_csv("AMZN","2015-09-30")
	#ya.make_csv("MSFT","2015-09-30")
	#ya.make_datemap("GOOG")

#	ya.make_csv("GOOG","2015-09-30")
	output=ya.get_data("GOOG")
#	print output["aver"][:10]



	#data=ya.get_data(Share("GOOG"),"2015-09-07")

	"""
	ya=Yahoo_aggregator()
	data=ya.get_data(Share("YHOO"),"2000-09-01","2000-09-07")
	print data["var_high"]
	"""

"""
MEMO


		self.csv_check(name,"2015-09-30")			#csvの存在チェック
#		start=share.get_info()["start"]

		f=open(self.DIR_NAME+name+".csv","r")
		print "reading "+name+".csv"
		reader = csv.DictReader(f)
		obj=[]
		for row in reader:
			obj.append(row)

		obj.reverse()


		#f=open("kaiai.csv","r")

"""







