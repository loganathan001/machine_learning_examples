#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:02:02 2019

@author: loganathan001
"""
import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N,D))

X[:,0] = 1
X[:5,1] = 1
X[5:,2] = 1

#target
Y = np.array([0]*5 + [1]*5)

# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# Error - Singular Matrix


costs = []

w = np.random.randn(D) / np.sqrt(D)

learning_rate = 0.001
epochs = 1000

for i in range(epochs):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * X.T.dot(delta)
    
    mse = delta.dot(delta) / N
    costs.append(mse)
    
plt.plot(costs)
plt.show()

print("w: ", w)

plt.plot(Yhat, label='predictions')
plt.plot(Y, label='target')
plt.legend()
plt.show()




