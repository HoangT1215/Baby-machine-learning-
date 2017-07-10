import numpy as np
import pandas as pd
import rpy2.robjects as robjects

#First, we gotta load the RData file weekly.rda
robjects.r['load']('data/weekly.RData')

matrix = robjects.r['fname']

# turn the R matrix into a numpy array
a = np.array(matrix)

print a