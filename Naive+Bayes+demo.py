
# coding: utf-8

# In[ ]:

#Tryout some Naive Bayes
import random
from sklearn.naive_bayes import GaussianNB

size = 20
k = 15

#data-gen
x = []
y = []

for i in range(size):
    a = random.randint(0,k)
    b = random.randint(0,k)
    while (a,b) in x:
        a = random.randint(0,k)
        b = random.randint(0,k)
    x.append([a,b])
    y.append(random.randint(0,1))

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])
print predicted

#Interestingly, the output return the same value of the pair