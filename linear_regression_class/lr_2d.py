#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 01:52:17 2019

@author: loganathan001
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X=[]
Y=[]

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))
    

X = np.array(X)
Y = np.array(Y)

#plot
figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

ax.scatter(X[:,0],X[:,1],Y)
plt.show()

#calc weights
w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

Yhat = X.dot(w)

#r-squard
diff_res = Y - Yhat
diff_mean = Y - Y.mean()

sse_res = diff_res.dot(diff_res)
sse_tot = diff_mean.dot(diff_mean)

r_squard = 1 - (sse_res/sse_tot)

print("RQuared: %f" % r_squard)
    