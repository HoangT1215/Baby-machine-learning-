from __future__ import division
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from sklearn import preprocessing
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

#read data
boston = pd.read_csv('../Data/Boston.csv')
print(boston.columns)

#plotting
def plotboston(df1,df2):
	plt.scatter(df1,df2)
	plt.show()

def corrtable():
	corr = boston.corr()
	sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
	sns.plt.show()

#applying models
def knearest():
	pass

def LDA():
	pass

#--- main program
med = np.median(boston['crim'])
print 'Median crime:', med
corrtable()

