from sklearn import tree
import pandas as pd
import numpy as np
from numpy import inf
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus

class Data():	#load data from cvs files and meagre them in to one data frame 
	df = pd.read_csv('breaset-cancer.csv', names = ['COUNTRY','INCOMEPERPERSON','ALCCONSUMPTION',
	'ARMEDFORCESRATE','BREASTCANCERPER100TH', 'CO2EMISSIONS','FEMALEEMPLOYRATE','HIVRATE','INTERNETUSERATE','LIFEEXPECTANCY','OILPERPERSON','POLITYSCORE','RELECTRICPERPERSON',
	'SUICIDEPER100TH','EMPLOYRATE','urbanrate'],encoding='latin-1') 
	df=df.fillna(df.mean())
	
	
	#separate the target and re evaluate column according to the rules
	tar = df.loc[:,'BREASTCANCERPER100TH']
	y = pd.cut(tar, [-inf,20, inf],labels=[0,1])
	
	#create a data-frame including only several attributes which are use to predict breast cancer percentage
	X = df[['ALCCONSUMPTION', 'CO2EMISSIONS','FEMALEEMPLOYRATE','INTERNETUSERATE','urbanrate']]
	
	
	# categorize  attributes by finding proper thresholds by looking at data set
	x1 = X.loc[:,'ALCCONSUMPTION']
	X.loc[:,'ALCCONSUMPTION'] = pd.cut(x1, [-inf,3, 6.67,inf],labels=[0,1,2])
	
	x2 = X.loc[:,'CO2EMISSIONS']
	X.loc[:,'CO2EMISSIONS'] = pd.cut(x2, [-inf,143000, 143000000,inf],labels=[0,1,2])
	
	x3 = X.loc[:,'FEMALEEMPLOYRATE']
	X.loc[:,'FEMALEEMPLOYRATE'] = pd.cut(x3, [-inf,35, 50,inf],labels=[0,1,2])
	
	x4 = X.loc[:,'INTERNETUSERATE']
	X.loc[:,'INTERNETUSERATE'] = pd.cut(x4, [-inf,15, 40,inf],labels=[0,1,2])
	
	x4 = X.loc[:,'urbanrate']
	X.loc[:,'urbanrate'] = pd.cut(x4, [-inf,40, 60,inf],labels=[0,1,2])
	
	#now all the attributes of the data-frame have categorized
	print("categorised data  :")
	print(X)
	
	
	#train a DecisionTreeClassifier model
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3333,random_state=0) 
	clf = tree.DecisionTreeClassifier()
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_val)#predict y values relevant to x values which are split to test
	y_all_pred=clf.predict(X)#predict y values relevant to all x values of the data set
	#print(y_pred)
	print("predicted breast cancer percentages :")
	print(y_all_pred)
	
	#checking the accuracy by using x values of all instances of the data set
	print("curacy of the model :" ,accuracy_score (y, y_all_pred ))
	
	'''
	#print a tree as pdf if want
	dot_data = tree.export_graphviz (clf , out_file =None)
	graph = pydotplus . graph_from_dot_data ( dot_data )
	graph.write_pdf ("breastcanser.pdf")
	'''
	
	
	
	
	
	'''
	
	For predicting breast cancer  percentage I used only following columns by looking at attributes how much relevant to the breast cancers
	'ALCCONSUMPTION', 'CO2EMISSIONS','FEMALEEMPLOYRATE','INTERNETUSERATE','urbanrate'
	(even OILPERPERSON also relevant to breast cancer that attribute has lot of missing values.it may cause to reduce accuracy of the model.
	Therefore it was neglected  )
	
	Then selected attribute values were categorize by defining threshold values by looking at data set and considering mean values of columns.defined thresholds are as follows
		ALCCONSUMPTION		alc <=3     -> 0
						    3<alc<=6.5  -> 1
						 	6.5<=alc    -> 2
							
		CO2EMISSIONS		co2 <=143000          -> 0
						    143000<co2<=14300000  -> 1
						 	14300000<=co2         -> 2
							
							
		FEMALEEMPLOYRATE	Femp <=35    -> 0
						    35<Femp<=50  -> 1
						 	50<=Femp     -> 2
								
							
		INTERNETUSERATE		int <=15    -> 0
						    15<int<=40  -> 1
						 	40<=int     -> 2
					
							
		urbanrate  			urb <=40     -> 0
						    40<urb<=60   -> 1
						 	60<=urb      -> 2
							
	Although I define these threshold values just by looking at data it should be done  by doing proper statistical analysis on the data set. 
	Because they affect to the accuracy of the model			
	
	'''
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
