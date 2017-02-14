#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 22:48:23 2017

@author: namratagannu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

creditData = pd.read_csv("/Users/namratagannu/Documents/DataScience/ML and FD/Datasets/credit_data.csv")

features = creditData [["income", "age", "loan"]]
targetVariables = creditData.default

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size = 0.3)

model = GaussianNB()
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest,predictions))
print(accuracy_score(targetTest, predictions))
