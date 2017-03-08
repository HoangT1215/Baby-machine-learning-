#Understanding PCA
import numpy as np
import sk-learn

#First we will generate 3D vectors sample
np.random.seed(234234782384239784) # random seed for consistency

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #covariance matrix

#covariance matrix is a matrix whose [i,j] value is the covariance between i-th and j-th elements of a random vector
#covariance is the joint variability of two random variables
