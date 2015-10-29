#coding:utf-8
import csv
from pprint import pprint
import numpy as np
import os



class CsvHandler:

	def __init__(self):
		self.DIR_NAME="../data/"
		print "save path:",self.DIR_NAME
		pass

	def csv_check(self,name,now_time):
	    if not os.path.exists(self.DIR_NAME+name+".csv"):
	    	self.make_csv(name,now_time)

	def make_csv(self,name,now_time):
		print "making %s.csv....." %name
		from yahoo_finance import Share
		share=Share(name)		
		start=share.get_info()["start"]		
		obj=share.get_historical(start,now_time)
		
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

	def make_datemap(self,name):
		obj = self.read_csv(name)
		g=open(self.DIR_NAME+name+"_datemap.csv","w")
		fieldnames=("Key","Date")
		headers = dict( (n,n) for n in fieldnames )
		writer=csv.DictWriter(g,fieldnames)	
		writer.writerow(headers)
		count = 1
		for o in obj:
			row=""
			writer.writerow({"Column":count,"Date":o["Date"]})
			count+=1
			
		g.close()



	def read_csv(self,name):
		self.csv_check(name,"2015-09-30")			#csvの存在チェック
		f=open(self.DIR_NAME+name+".csv","r")
		print "reading "+name+".csv"
		reader = csv.DictReader(f)
		obj=[]
		for row in reader:
			obj.append(row)
		obj.reverse()
		f.close()
		return obj		


	#yahoo_financeから
	#obj=share.get_historical(start,now_time)
	#csvからよみこみ
	def get_data(self,name):
		obj = self.read_csv(name)

		date = []
		date.append(obj[0]["Date"])

		high=np.array([],dtype="float32")
		low=np.array([],dtype="float32")
		vol=np.array([],dtype="float32")
		aver=np.array([],dtype="float32")

		before_high=float(obj[0]["High"])
		before_low=float(obj[0]["Low"])
		before_vol=float(obj[0]["Volume"])

		for day in obj[1:]:

			date.append(day["Date"])
			aver=np.append(aver,(float(day["High"])+float(day["Low"]))/2)
			high=np.append(high,(float(day["High"])-before_high)/before_high)
			low=np.append(low,(float(day["Low"])-before_low)/before_low)
			vol=np.append(vol,(float(day["Volume"])-before_vol)/before_vol)

			before_high=float(day["High"])
			before_low=float(day["Low"])
			before_vol=float(day["Volume"])

		output={"start":date[0],"date":date,"high":high,"low":low,"vol":vol,"aver":aver}
		return output




