from yahoo_finance import Share
from pprint import pprint
import matplotlib.pyplot as plt



class Company:

	def __init__(self,name,d_data):
		self.name=name
		self.d_data=d_data

	def plot(self,start_day,end_day,filename="graph"):
		share=Share(self.name)
		obj=share.get_historical(start_day,end_day)


		x=[]
		y_high=[]
		y_low=[]

		i=1
		

		for day in obj:
			x.append(i)
			y_high.append(float(day['High']))
			y_low.append(float(day['Low']))
			i+=1

		plt.plot(y_high)
		plt.plot(y_low)
		#plt.show()
		#plt.savefig(filename+".jpg")

if __name__=="__main__":
	
	filename="data/companylist.csv"	
	i=0
	for line in open(filename,"r"):
		if i==0:
			i+=1
			continue
		
		name=line.split(",")[0]
		name=name.replace("\"","")
		
		share=Share(name)
		print share.get_price()

	#google.plot("2014-07-20","2015-09-30","google")








