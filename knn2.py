#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 19:20:38 2017

@author: namratagannu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

creditData = pd.read_csv('/Users/namratagannu/Documents/DataScience/ML and FD/Datasets/credit_data.csv')

features = creditData [['income','age','loan']]
targetVariables = creditData.default

#IMPORTANT STEP!!
features = preprocessing.MinMaxScaler().fit_transform(features)

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size = 0.3)

model = KNeighborsClassifier(n_neighbors = 4)  #This is the k value, you can change it around
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest,predictions))
print(accuracy_score(targetTest, predictions))

#for logistic regression, accuracy was 93%
# for knn, accuracy is 97.5%.  We do better with KNN