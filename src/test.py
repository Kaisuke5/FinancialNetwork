from Company import Company
from YahooAggregator import yahoo_aggregator


name="AMZN"
cp=Company(name)

#get data of yahoo finance from csv
ya=yahoo_aggregator()
cp.get_data(ya)


#make train_data
K,term,per=30,30,0.10
output=cp.make_train_data(term,K,per)
cp.make_pikle(output)