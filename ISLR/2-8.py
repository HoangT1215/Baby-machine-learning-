#statistical learning, problem 2-8
import numpy as np
import pandas as pd
import csv

#a)Read the file college.csv
df = csv.reader('College.csv')
df.fillna(value=-99999, inplace=True)

#b)Reconfig the data

#