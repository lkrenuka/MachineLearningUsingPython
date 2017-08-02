#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#features_train = features_train[:round(len(features_train)/100)]
#labels_train = labels_train[:round(len(labels_train)/100)] 
#print(features_test.shape)
clf = SVC(kernel='rbf', C=10000)
t0=time()
clf.fit(features_train, labels_train)
print("Training time in secs")
print(round(time()-t0))
t1=time()
pred = clf.predict(features_test)
print("Prediction time in secs")
print(round(time()-t1))
chs = np.count_nonzero(pred)
sar = len(pred) - chs
print ("Chris emails....")
print (chs)
print ("Sara emails....")
print (sar)
#accuracy = accuracy_score(pred, labels_test)
#print (accuracy)
