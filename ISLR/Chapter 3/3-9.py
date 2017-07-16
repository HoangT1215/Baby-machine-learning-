from __future__ import division
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from pandas import scatter_matrix
import matplotlib.pyplot as plt

#read data
auto = pd.read_csv('../Data/Auto.csv')
df = auto.drop('name',1)
'''
Columns
Index([u'mpg', u'cylinders', u'displacement', u'horsepower', u'weight',
       u'acceleration', u'year', u'origin', u'name'],
      dtype='object')
'''

def plotting(df1,df2):
	pass

def task_a(data):
	scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')


def task_b(data):
	corr = data.corr()
	sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

def task_c():
	pass



#--- main program
task_a(df)
task_b(df)