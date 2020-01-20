#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 07:16:47 2019

@author: loganathan001
"""
from sklearn.linear_model.base import LinearRegression

"""
Denver Neighborhoods

The data (X1, X2, X3, X4, X5, X6, X7) are for each neighborhood
X1 = total population (in thousands)
X2 = % change in population over past several years
X3 = % of children (under 18) in population
X4 = % free school lunch participation
X5 = % change in household income over past several years
X6 = crime rate (per 1000 population)
X7 = % change in crime rate over past several years
Reference: The Piton Foundation, Denver, Colorado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dfgui


from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_squared_error

df = pd.read_excel('mlr10.xls')

df.columns = ['total_population', 
              'pt_ch_in_population',
              'pt_of_children_und18',
              'pt_free_school_lunch_partpn',
              'pt_ch_in_household_income',
              'crime_rate_per1000',
              'ch_in_crime_rate']

#g = sns.pairplot(df)

#plt.show()

#dfgui.show(df)
# view correlation
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()

# ElasticNet Cross validation
reg = ElasticNetCV(l1_ratio=1.0, 
                    #alphas=np.array((0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100)), 
                    normalize=True, max_iter=50, 
                    #tol=0.3, 
                    cv=8, verbose=True, 
                    fit_intercept = True,
                    random_state=100,
                    n_jobs=4)

#Linear Regression
# reg = LinearRegression(normalize=True)

target_col = 'crime_rate_per1000'
feature_cols = list(df.columns)
feature_cols.remove(target_col)
X = df[feature_cols]
y = df[target_col]

train_errors = []
test_errors = []
epochs = 100

for e in range(epochs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg.fit(X_train, y_train)

    for i in range(X_train.shape[0]):
        train_errors.append(mean_squared_error(y_train, reg.predict(X_train)))
        
    for i in range(X_test.shape[0]):
        test_errors.append(mean_squared_error(y_test, reg.predict(X_test)))

plt.plot(train_errors, label = 'train errors')
plt.plot(test_errors, label = 'test errors')

plt.legend()

plt.show()







