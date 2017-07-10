import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.lda import LDA

boston = pd.read_csv('../Data/Boston.csv')
print(boston.columns)

row = boston.shape[0]
column = boston.shape[1]

print(boston['medv'])

def task_a():

	pass

