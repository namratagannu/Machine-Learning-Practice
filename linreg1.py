#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:45:09 2017

@author: namratagannu
"""

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("/Users/namratagannu/Documents/DataScience/ML and FD/Datasets/linreg.csv")

npMatrix = np.matrix(dataset)
X,Y = npMatrix[:,0], npMatrix[:,1]

regression = linear_model.LinearRegression()

#why is the coef 0?  (runs even without the [0])

regression.fit(X,Y)
a = regression.intercept_
b = regression.coef_[0]
 
print ("The coefficient is 'b' \n ", b)
print ("The intercept is 'b' \n ", a)

print 'H(x) = ',a,' + ',b,'*x'

plt.scatter(X,Y, color = 'black')
plt.plot(X,regression.predict(X), color = 'blue', linewidth = 3)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)


print 'X= 500, Y =',regression.predict(500)
plt.show()

