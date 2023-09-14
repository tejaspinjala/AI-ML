#!/usr/bin/env python
# coding: utf-8

# In[2]:


# an attempt to understand what goes on behind the sklearn LogisticRegression class
import numpy as np

def sigmoid(x): # activation function
    return 1/(1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters = 1000): # epoch in Nural Network or Deep Learning
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y): # get the best coefficients and bias
        n_samples, n_features = X.shape # samples = m, features = n
        self.weights = np.zeros(n_features) # creates an array
        self.bias = 0 
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias # dot prodcut(X = feautes. Transpose matrix). y = mx + b with multiple w's
            y_pred = sigmoid(y_pred)
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y)) # partial derivative with respect to w
            db = (1/n_samples) * np.sum(y_pred-y) # partial derivative with respect to b
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    def predict(self, X): # predict once we have the best coefficients and bias
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(y_pred)
        prediction = [0 if y <= 0.5 else 1 for y in y_pred]
        return prediction


# In[ ]:




