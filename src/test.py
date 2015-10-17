from Company import Company
from YahooAggregator import Yahoo_aggregator


name="FB"
cp=Company(name)

#get data of yahoo finance from csv
Yahoo = Yahoo_aggregator()
cp.get_data(Yahoo)


#make train_data
#K,term,per=30,30,0.10
#output=cp.make_train_data(term,K,per)
#cp.make_pickle(output)
