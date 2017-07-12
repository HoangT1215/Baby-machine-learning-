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
'''

#read data
boston = pd.read_csv('../Data/Boston.csv')
print(boston.columns)

#plotting
def plotboston():
	pass

#applying models
def knearest():
	pass

def LDA():
	pass