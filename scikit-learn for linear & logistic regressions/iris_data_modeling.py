from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,3]
X=X.reshape((-1, 1))
y = iris.target
		
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

# train a logistic regression model 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) 
log_reg=LogisticRegression() 
log_reg.fit(X_train,y_train)
y_proba=log_reg.predict(X_val)
accuracy_logi = log_reg.score(X_val, y_val)
print("predicted y values of X_val is(logistic regression):",y_proba)
print("accuracy (logistic regression):",accuracy_logi)




from sklearn.linear_model import LinearRegression 
# train a linear regression model 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) 
lin_reg=LinearRegression() 
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_val)
accuracy_lin = lin_reg.score(X_val, y_val)
print("predicted y values of X_val is:(linear regression)",y_proba)
print("accuracy of (linear regression) :",accuracy_lin)

