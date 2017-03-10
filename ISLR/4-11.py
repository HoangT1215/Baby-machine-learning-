import pandas as pd
import numpy as np

#read data
auto = pd.read_csv('Data/Auto.csv')
print(auto.columns)
'''
Index([u'mpg', u'cylinders', u'displacement', u'horsepower', u'weight',
       u'acceleration', u'year', u'origin', u'name'],
      dtype='object')
'''
#get number of row from the dataframe
row = auto.shape[0]
column = auto.shape[1]

#drop unnecessary columns
df = auto.drop('origin', 1)
df = df.drop('name', 1)

#task a

def task_a():
	mpg01 = []
	mpg = auto['mpg']
	mean_mpg = np.mean(mpg)
	for i in range(row):
		if (mpg[i] < mean_mpg):
			mpg01.append(0)
		else:
			mpg01.append(1)
	print mpg01

#task b
#approach the task with LogReg
