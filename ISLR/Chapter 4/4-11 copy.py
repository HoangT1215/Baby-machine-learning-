from __future__ import division
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
mpg = df['mpg']
horse_power = df['horsepower']
horse_power = horse_power.convert_objects(convert_numeric=True) # convert to float64

def plotauto(df1,df2):
	# plot the data
	plt.scatter(df1,df2)
	plt.show()

def accuracy(dat):
	match = 0
	for i in range(len(dat)):
		if dat[i] == mpg01[i]:
			match += 1
	acc = float(match/len(dat))
	return acc

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
	print(corr)
	sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
	# reference: https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas

'''
task b findings:
mpg is most correlated with weight, displacement and cylinders
'''

def task_c(data):
	# splitting test data and training data
	auto_test, auto_train = train_test_split(data, test_size = 0.2, random_state = 42)
	return (auto_test,auto_train)
	
def task_d(train,test):
	clf = LinearDiscriminantAnalysis()
	x = np.array(train)
	y = np.array(test)
	clf.fit(x,trainmpg01)
	result = clf.predict(y)
	error = 0
	for i in range(len(testmpg01)):
		if (testmpg01[i] != result[i]):
			error += 1
	print(error/len(result))

def task_e(train,test):
	clf = QuadraticDiscriminantAnalysis()
	x = np.array(train)
	y = np.array(test)
	clf.fit(x,trainmpg01)
	result = clf.predict(y)
	error = 0
	for i in range(len(testmpg01)):
		if (testmpg01[i] != result[i]):
			error += 1
	print(error/len(result))


def task_f():
	# perform logistics regression to predict mpg01 and get test error

	pass

def train(dat):
	pass

#--- main program
df['mpg01'] = task_a()
data_d = df.drop('mpg',1)
data_d = data_d.drop('acceleration',1)
data_d = data_d.drop('horsepower',1)
test_data, train_data = task_c(data_d)
trainmpg01 = np.array(train_data['mpg01'])
testmpg01 = np.array(test_data['mpg01'])
task_d(train_data,test_data)
task_e(train_data,test_data)