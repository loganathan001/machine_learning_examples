#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:02:02 2019

@author: loganathan001
"""


#Optimizing J = w1**2 + w2**4
#dj/dw1 = 2w
#dj/dw2 = 4*w2**3
# w <- lr * w - dj/d2 <- w- lr * 2w
lr = 0.0001
w1 = 20
w2 = 40
ephocs = 10000
for i in range(ephocs):
    w1 = w1 - lr * 2 * w1
    w2 = w2 - lr * 4 * w2**3
    J = w1**2 + w2**4
    print(J)
    
print("Final w:1 ", w1, " and w2:",  w2)


