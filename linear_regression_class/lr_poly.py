#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:40:18 2019

@author: loganathan001
"""

import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_poly.csv'):
    x,y = line.split(',')
    
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

#plot

plt.scatter(X[:, 1], Y)
plt.show()

#weights
w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat = X.dot(w)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:,1]),sorted(Yhat))
plt.show()

#R squared
diff_res = Y - Yhat
diff_mean = Y - Y.mean()

sse_res = diff_res.dot(diff_res)
sse_tot = diff_mean.dot(diff_mean)

r_squard = 1 - (sse_res/sse_tot)

print("RQuared: %f" % r_squard)