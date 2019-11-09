#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:10:10 2019

@author: loganathan001
"""

# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

df['ones'] = 1
X = df[['X2','X3','ones']]
Y = df['X1']

X2Only = df[['X2', 'ones']]
X3Only = df[['X3', 'ones']]

def get_rsquared(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)
    diff_res = Y - Yhat
    diff_mean = Y - Y.mean()
    
    sse_res = diff_res.dot(diff_res)
    sse_tot = diff_mean.dot(diff_mean)

    r_squard = 1 - (sse_res/sse_tot)
    return r_squard

print('RSquared for X2 only', get_rsquared(X2Only, Y))
print('RSquared for X3 only', get_rsquared(X3Only, Y))
print('RSquared for both', get_rsquared(X, Y))

#random noise in x
np.random.seed(100)
df['random'] = np.random.randint(1,101, size=df.shape[0])
XNoise = df[['X2','X3']].sort_values(axis=0,by=['X2'])
XNoise['random'] = df['random']

print('RSquared for X with random noise', get_rsquared(XNoise, Y))


