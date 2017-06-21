
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')

#setup data
dataset = {'k':[[1,2],
                [2,3],
                [3,1]],
           'r':[[6,5],
                [7,7],
                [8,6]]}#list in a list structure
new_features = [5,7]

#plotting
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)

plt.show()
#end plotting

def k_nearest_neighbors(data, predict, k=3): #train data, predict data, nearest k
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    
    distances = []
    for group in data: #go through all lists: list k and list r in dataset
        for features in data[group]: #features are the coordinates
            euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
            distances.append([euclidean_distance,group]) #append distance array, note that the group is linked to the distance
    
    votes = [i[1] for i in sorted(distances)[:k]] #sort and find the k nearest ones
    vote_result = Counter(votes).most_common(1)[0][0] #setup the counter in the array votes, return only the type

    return vote_result

result = k_nearest_neighbors(dataset, new_features)


# In[ ]:



