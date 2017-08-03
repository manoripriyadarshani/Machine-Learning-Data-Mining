import pandas as pd
import numpy as np
import unittest
from pandas.util.testing import assert_frame_equal


 
class Data():

	def __init__(self,df):
		self.df=df	
		
	
	#this method return covariance between given a and b columns
	def C(self,col_a,col_b):
	
		x=self.df.apply(np.mean)
		sum = 0
		# iterate through rows
		for j in range(len(self.df.index)):
			t1 = x.ix[col_a,j]
			t2 = x.ix[col_b,j]
			sum += (self.df.ix[j,col_a]-t1)*(self.df.ix[j,col_b]-t2)
					
		std_dev_ofacol = sum * 1 / (len(self.df.index)-1)
		return (std_dev_ofacol)
		
		
		
	#static method return variance of a given columns	
	def get_varince(self,col_a):
		y=self.df.var(axis=0)
		return y.ix[col_a,0]
		
		
		
		
	# instance method to find the covariance matrix	
	def Covariance_metrix(self):	
		
		Cov_matrix = pd.DataFrame()
		# iterate through columns
		for i in range(len(self.df.columns)):
			# iterate through rows
			for j in range(len(self.df.columns)):
				if i==j:
					Cov_matrix.set_value(j, i, self.get_varince(i))
					
				elif j>i:
					C_val=self.C(j,i)
					#fill the lower part element of the main diagonal
					Cov_matrix.set_value(j, i,C_val)
					#fill the upper part element of the main diagonal
					Cov_matrix.set_value(i, j,C_val)
		print ('*************** This is the covariance metrix ***************')		
		print (Cov_matrix)
		return (Cov_matrix)
		
		
	
			
			
	#This is the instance method to find the Correlation matrix				
	def Correlation_metrix(self):	
		
		y=self.df.var(axis=0)
		Cor_matrix=pd.DataFrame()
		# iterate through columns
		for i in range(len(self.df.columns)):
			# iterate through rows
			for j in range(len(self.df.columns)):
				if j>=i:
					r_val=self.C(j,i) / (np.sqrt(self.get_varince(i))*np.sqrt(self.get_varince(j)))
					#fill the lower part element of the main diagonal and elements on the diagonal
					Cor_matrix.set_value(j, i,r_val)
					#fill the upper part element of the main diagonal
					Cor_matrix.set_value(i, j,r_val)
		print ('*************** This is the corrilation metrix ***************')	
		print (Cor_matrix)
		return (Cor_matrix)
	

	
	
	
	
	
class Test(unittest.TestCase):	

	def setUp(self):
		#read data from csv file
		self.df1 = pd.read_csv('dataset.csv', names = ['a','b','c','d','e']) 
		#fill empty element with mean of each column
		self.df1=self.df1.fillna(self.df1.mean())
		#create a object for test given cvs file data set
		self.obj1=Data(self.df1)
		
		
		
		
	def test_Covariance_metrix(self):
	
	
		#test method for known matrix
		cc
		df2_actual_covariance =	pd.DataFrame({0: [4.0, -1.0, 3.0],1: [-1.0,1.0,-3.0],2: [3.0, -3.0, 9.0 ], })
		#create a object for test known data set
		obj2= Data(df2)
		assert_frame_equal(df2_actual_covariance, obj2.Covariance_metrix())
		
		
		#test method for given data set
		df1_actual_covariance = pd.DataFrame({0: [0.000588, 0.002601, 0.002420,0.001073,0.001369],1: [0.002601,0.016716,0.013902,0.006063,0.009845],2: [0.002420,0.013902, 0.015712,0.006316,0.010310 ],3: [0.001073, 0.006063,0.006316,0.003114,0.005489  ],4: [0.001369, 0.009845,0.010310, 0.005489 ,0.018272], })
		df1_method_covariance=self.obj1.Covariance_metrix()
		df1_method_covariance=df1_method_covariance.round(6)
		assert_frame_equal(df1_actual_covariance, df1_method_covariance)
	
		
		
		
			
	def test_Correlation_metrix(self):
	
	
		#test method for known matrix
		df2 = pd.DataFrame({0: [2, 6, 4],1: [5,4,6],2: [7, 10, 4 ], })
		df2_actual_Correlation =	pd.DataFrame({0: [1.0, -0.5, 0.5],1: [-0.5,1.0,-1.0],2: [0.5,  -1.0, 1.0 ], })
		obj2= Data(df2)
		df2_methodgiven_Correlation=obj2.Correlation_metrix()
		assert_frame_equal(df2_actual_Correlation, df2_methodgiven_Correlation.round(6))
		
		#test method for given data set
		df1_actual_Correlation = pd.DataFrame({0: [1.000000, 0.829808,  0.796262, 0.793288 ,0.417635],1: [0.829808,1.000000,0.857834,0.840377,0.563314],2: [ 0.796262,0.857834 , 1.000000,0.903041,0.608506 ],3: [0.793288, 0.840377, 0.903041,1.000000,0.727713  ],4: [0.417635 , 0.563314 ,0.608506 , 0.727713 ,1.000000], })
		df1_method_Correlation=self.obj1.Correlation_metrix()
		df1_method_Correlation=df1_method_Correlation.round(6)
		assert_frame_equal(df1_actual_Correlation, df1_method_Correlation)
		
		
		
		
		
		
if __name__ == '__main__':
	unittest.main()
	
	
	
