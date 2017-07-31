from __future__ import division
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

'''
Goal:
- Predict whether a given suburb has a higher crime rate than the median (binary classifying problem)
- Testing predictive power of models

Data:
Index([u'crim', u'zn', u'indus', u'chas', u'nox', u'rm', u'age', u'dis',
       u'rad', u'tax', u'ptratio', u'black', u'lstat', u'medv'],
      dtype='object')
'''

#read data and preprocess
boston = pd.read_csv('../Data/Boston.csv')
print(boston.columns)

#get number of row from the dataframe
row = boston.shape[0]
column = boston.shape[1]

dat = boston.drop('indus',1)
dat = dat.drop('chas',1)
dat = dat.drop('ptratio',1)
crim = dat['crim']

#plotting
def plotboston(df1,df2):
	plt.scatter(df1,df2)
	plt.show()

def clf_med():
	medv = []
	for i in range(row):
		if (crim[i] < med):
			medv.append(0)
		else:
			medv.append(1)
	return medv

def corrtable():
	corr = dat.corr()
	sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
	plt.show()

#split
def split(data):
	# splitting test data and training data
	auto_test, auto_train = train_test_split(data, test_size = 0.2, random_state = 42)
	return (auto_test,auto_train)

#applying models
def knearest(train,test):
	knn = neighbors.KNeighborsClassifier()
	train = np.array(train)
	test = np.array(test)
	knn.fit(train,medv01)
	result = knn.predict(test)
	error = 0
	for i in range(len(medv01)):
		if (medv01[i] != result[i]):
			error += 1
	print(error/len(result))

def LDA(train,test):
	linda = LinearDiscriminantAnalysis()
	train = np.array(train)
	test = np.array(test)
	linda.fit(train,medv01)
	result = linda.predict(test)
	error = 0
	for i in range(len(medv01)):
		if (medv01[i] != result[i]):
			error += 1
	print(error/len(result))

#--- main program
med = np.median(boston['crim'])
medv01 = clf_med()
print('Median crime:', med)
testdat, traindat = split(dat)
knearest(traindat,testdat)
LDA(traindat,testdat)

