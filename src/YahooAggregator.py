#coding:utf-8
""" 
Read from Yahoo-finance and make csv file

Kai & Shiba

Last Stable:
2015/10/16

Last updated:
2015/10/16
"""
from yahoo_finance import Share
import csv
from pprint import pprint
import os


class Yahoo_aggregator:

	def __init__(self):
		self.DIR_NAME="../data/"
		pass

	def csv_check(self,name):
	    if not os.path.exists(name):
	    	print "making new csv..."
	        self.make_csv(name,"2015-09-30")

	def make_csv(self,name,now_time):
		print "making %s.csv....." %name
		share=Share(name)
		
		start=share.get_info()["start"]
		
		obj=share.get_historical(start,now_time)
		
		start=share.get_info()["start"]
		filename=name+".csv"
		fieldnames=("Date","High","Low","Volume" )
		headers = dict( (n,n) for n in fieldnames )
		f=open(self.DIR_NAME+filename,"w")
		writer=csv.DictWriter(f,fieldnames)	
		writer.writerow(headers)

		for o in obj:
			row=""
			writer.writerow({"Date":o["Date"],"High":o["High"],"Low":o["Low"],"Volume":o["Volume"]})
			
		f.close()


	#yahoo_financeから
	#obj=share.get_historical(start,now_time)
	#csvからよみこみ
	def get_data(self,share,name):
		self.csv_check(self.DIR_NAME+name+".csv")			#csvの存在チェック
		start=share.get_info()["start"]

		f=open(self.DIR_NAME+name+".csv","r")
		print "reading "+name+".csv"
		reader = csv.DictReader(f)
		obj=[]
		for row in reader:
			obj.append(row)


		obj.reverse()
		var_high, var_low, var_vol, aver_price=[],[],[],[]


		before_high=float(obj[0]["High"])
		before_low=float(obj[0]["Low"])
		before_vol=float(obj[0]["Volume"])

		for day in obj[1:]:
			aver_price.append((float(day["High"])+float(day["Low"]))/2)
			var_high.append((float(day["High"])-before_high)/before_high)
			var_low.append((float(day["Low"])-before_low)/before_low)
			var_vol.append((float(day["Volume"])-before_vol)/before_vol)

			before_high=float(day["High"])
			before_low=float(day["Low"])
			before_vol=float(day["Volume"])


		output={"start":start,"var_high":var_high,"var_low":var_low,"var_vol":var_vol,"aver_price":aver_price}
		return output






if __name__=="__main__":
	ya=Yahoo_aggregator()
	ya.make_csv("GOOG","2015-09-30")
	ya.make_csv("AMZN","2015-09-30")
	ya.make_csv("YHOO","2015-09-30")
	

	#data=ya.get_data(Share("GOOG"),"2015-09-07")

	"""
	ya=Yahoo_aggregator()
	data=ya.get_data(Share("YHOO"),"2000-09-01","2000-09-07")
	print data["var_high"]
	"""

"""
MEMO

		#f=open("kaiai.csv","r")

"""







