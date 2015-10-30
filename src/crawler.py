import CsvHandler 
import errno
from socket import error as socket_error
import time
import csv

C=CsvHandler.CsvHandler()
DATE="2015-10-28"
companys=["FB","GOOG"]
sleep_time=60
NUM=10


count=0
companys=[]
for line in open("../source/companylist.csv","r"):
	if count==0: pass
	elif count>NUM:break
	else:
		a=line.split(",")[0][1:-1]
		companys.append(a)

	count+=1



print companys


for company in companys:
	
	while True:
		try:
		    C.make_csv(company,DATE)
		    break
		except socket_error as serr:
				print "sleeping"
				time.sleep(sleep_time)
				continue
