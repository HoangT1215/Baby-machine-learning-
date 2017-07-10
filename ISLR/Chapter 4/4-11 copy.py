import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

#read data
auto = pd.read_csv('../Data/Auto.csv')
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
mpg = auto['mpg']
horse_power = auto['horsepower']
horse_power = horse_power.convert_objects(convert_numeric=True) # convert to float64

def task_a():
	mpg01 = []
	median_mpg = np.median(mpg)
	for i in range(row):
		if (mpg[i] < median_mpg):
			mpg01.append(0)
		else:
			mpg01.append(1)
	return mpg01

#task b
#approach the task with LogReg
def task_b():
	# cross correlation
	print("Pearson's r value:", mpg.corr(horse_power))
	# correlation matrix
	corr = df.corr() # df is from auto
	sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
	# reference: https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas

	# plot the data
	plt.scatter(mpg,horse_power)
	plt.xlabel('Gas mileage')
	plt.ylabel('Horse power')
	plt.show()

	# assess accuracy

def task_c():
	auto_test, auto_train = train_test_split(df, test_size = 0.2, random_state = 42)
	
def task_d():
	clf = LinearDiscriminantAnalysis()
	pass

def task_e():
	pass

def task_f():
	pass

def train(dat):
	pass

def accuracy(dat):
	match = 0
	for i in range(len(dat)):
		if dat[i] == mpg01[i]:
			match += 1
	acc = float(match/len(dat))
	return acc

#--- main program
