#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:24:08 2019

@author: loganathan001
"""

import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)

Y[-1] += 30
Y[-2] += 30

plt.scatter(X,Y)
plt.show()

#Adding bias term
X = np.vstack([np.ones(N), X]).T

#maximul liklihood solution
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(X))
Yhat_ml = X.dot(w_ml)

plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml)
plt.show()

#l2 regularization solution
#l2 penalty
l2 = 1000.0
#MAximizeing Posterior
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)

plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:, 1], Yhat_map, label='map (l2 regularized')
plt.legend()
plt.show()