#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:07:39 2019

@author: loganathan001
"""

import re
import numpy as np
import matplotlib.pyplot as plt

X=[]
Y=[]

non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')
    
    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    
    X.append(x)
    Y.append(y)
    
X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.show()

Y = np.log(Y)
plt.scatter(X,Y)
plt.show()

denominator =  (X**2).mean() - X.mean()**2

a = ((X*Y).mean() - X.mean()*Y.mean())/denominator
b = (Y.mean()*(X**2).mean() - X.mean()*(X*Y).mean())/denominator

Yhat = a*X + b

plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

# Calculate R**2
diff_res = Y - Yhat
diff_mean = Y - Y.mean()

sse_res = diff_res.dot(diff_res)
sse_tot = diff_mean.dot(diff_mean)

r_squard = 1 - (sse_res/sse_tot)

print('a', a, 'b', b)
print("RQuared: %f" % r_squard)


#log(transister_count) = a*year + b
#  transister_count = exp(a*year) * exp(b)
# 2 * transister_count  = 2 * exp(a*year) * exp(b)
#                       = exp(ln(2)) * exp(a*Xyear * exp(b)
#                       = exp(a*year + ln(2)) * exp(b)
#year2 -> the year to double
#  exp(a*year2) * exp(b) = exp(a*year1 + ln(2)) * exp(b)
#  a*year2 = a*year1 + ln(2)
#  year2 = year1 + ln(2)/a
#  time_to_doauble = year2 - year1 = ln(2)/a
#
print("time to double" , np.log(2)/a)