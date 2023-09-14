#!/usr/bin/env python
# coding: utf-8

# $$
# \begin{align}
# \frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
# \frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
# \end{align}
# $$
# * m is the number of training examples in the data set
# 
#     
# *  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value

# In[2]:


# an attempt to understand what goes on behind the sklearn LinearRegressioin class
import numpy as np

class LinearRegression:
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
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y)) # partial derivative with respect to w
            db = (1/n_samples) * np.sum(y_pred-y) # partial derivative with respect to b
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    def predict(self, X): # predict once we have the best coefficients and bias
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


# In[ ]:




