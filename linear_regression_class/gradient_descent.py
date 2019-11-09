#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:02:02 2019

@author: loganathan001
"""


#Optimizing J = w**2
#dj/dw = 2w
# w <- lr * w - dj/d2 <- w- lr * 2w
lr = 0.1
w = 20
ephocs = 100
for i in range(ephocs):
    w = w - lr * 2 * w
    J = w**2
    print(J)
    
print("Final w: ", w)

