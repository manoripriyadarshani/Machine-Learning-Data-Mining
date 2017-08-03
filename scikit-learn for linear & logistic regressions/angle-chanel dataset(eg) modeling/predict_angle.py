import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score


class Data():

	#load data from cvs files and meagre them in to one data frame 
	df1 = pd.read_csv('Channels.csv', names = ['channel1','channel2','channel3','channel4','channel5']) 
	df2 = pd.read_csv('Angles.csv', names = ['angle1','angle2','angle3']) 
	df3=pd.concat([df1, df2], axis=1)
	
	
	#set feature attribute and predict attribute
	X = df3[['channel1', 'channel2','channel3','channel4','channel5']]
	y = np.array(df3[['angle1']])
	y = y.ravel()
	
	
    #train a leaner regression model
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) 
	lin_reg=LinearRegression() 
	lin_reg.fit(X_train,y_train)
	y_proba=lin_reg.predict(X_val)
	y_pred=lin_reg.predict(X)
	
	#checking the accuracy by inputting all x instances that we have  
	print("curacy of the model :" ,r2_score (y_val, y_proba ))

	
	#predict angle value for known channel values as a example 
	print("applying model to a new instance....")
	x_new= pd.DataFrame({0: [-0.052098],1:[-0.235460] ,2:[-0.150474],3:[-0.024574],4:[0.026625]})
	y_pred_new=lin_reg.predict(x_new)
	print("angle value for channel values of a new instance: ", y_pred_new)
	
	
	
	
'''
	
	(3)
	In linear regression, the outcome (dependent variable) is continuous. It can have any one of an infinite number of possible values. 
	In logistic regression, the outcome (dependent variable) has only a limited number of possible values.
	Logistic Regression is used when response variable is categorical in nature. For instance, Yes/No, True/False, Red/Green/Blue, etc. 
	Linear Regression is used when your response variable is continuous. For instance Weight, Height, Number of hours etc.
	In this example we are going to predict angle .I also continuous field. Therefore linear regression is most suitable 

	(4)
	selecting more feature attributes is better than considering only one attribute and just neglecting other attributes.
	It cause to increase the accuracy of the predictions
	Therefore I used all channels as features and  one angle data for predicting
	(angle2 was predicted using channel1,channel2,channel3,channel4,channel5 )
	
	(5)
	Accuracy of these model is 4.8% .Where data set has splited 20% for testing and 80% for training
	When changing test_size ,the accuracy of the model was changed.
	When test_size make increase accuracy become low and also When test_size make decrease accuracy become high.
	Because the model does train well,when increase number of instances that used to train data  
	
'''	

	
	

	
	
