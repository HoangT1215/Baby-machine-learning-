'''
The main purposes of a principal component analysis (PCA) are the analysis of data to identify patterns 
and finding patterns to reduce the dimensions of the dataset with minimal loss of information.

Here, we will play around with 3-D vectors and reduce it to 2-D to see how PCA works
'''

#Understanding PCA
import numpy as np
import sk-learn
import matplotlib.pyplot as plt

#First we will generate 3D vectors sample
np.random.seed(234234782384239784) # random seed for consistency

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #covariance matrix

#covariance matrix is a matrix whose [i,j] value is the covariance between i-th and j-th elements of a random vector
#covariance is the joint variability of two random variables

class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

#Created 2 databases, each has 20 3-D vectors

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

#class 2, also have 20 3D-vectors

#Step 1, taking the whole dataset ignoring the class label
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3,40), "The matrix has not the dimensions 3x40"

#Step 2, compute the mean d-D vector
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('Mean Vector:\n', mean_vector)

#Step 3a, compute the scatter matrix
scatter_matrix = np.zero(3,3)
n = all_samples.shape[1]
for i in range(n):
	scatter_matrix += (all_samples[:,i].reshape(3,1)-mean_vector).dot((all_samples[:,i].reshape(3,1)-mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)

#Step 3b, compute the covariance matrix
cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print('Covariance Matrix:\n', cov_mat)

#Step 4, computing eigenvalues
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)


#visualize
def vector_viz(vec_mat):
	plt.plot(vec_mat)
	plt.show()
