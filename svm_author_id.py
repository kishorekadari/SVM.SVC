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
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
#from matplotlib import style
#   style.use("gplot")

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
labels_train=(np.array(labels_train))
#labels_train = np.argmax(labels_train, axis=1)
#features_test=((features_test).reshape(-1,1))
labels_test=(np.array(labels_test))

#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf=svm.SVC(kernel='linear')
t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")
t1 = time()
y_pred= clf.predict(features_test)
print ("training time:", round(time()-t1, 3), "s")
accuracy= accuracy_score(y_pred, labels_test)
print (accuracy)
#########################################################


