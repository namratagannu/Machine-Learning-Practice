# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[10000,80000,35],[7000,120000,57],[100,23000,22],[223,18000,26]])
Y = np.array([1,1,0,0])

classifier = LogisticRegression()

classifier.fit(X,Y)

print(classifier.predict_proba([5500,50000,25]))