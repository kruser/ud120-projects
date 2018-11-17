#!/usr/bin/python

"""
This is the code to accompany the Lesson 2 (SVM) mini-project.

Use a SVM to identify emails from the Enron corpus by their authors:    
Sara has label 0
Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# cut down the features to speed up training
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

def train_and_predict(control=1):
    print('Training with control of %d' % control)

    start_time = time()
    classifier = SVC(kernel="rbf", C=control)

    classifier.fit(features_train, labels_train)
    print "Training time:", round(time()-start_time, 3), "s"

    prediction = classifier.predict(features_test)
    print "Prediction time:", round(time()-start_time, 3), "s"

    print(accuracy_score(prediction, labels_test))

    chris = 0
    sarah = 0
    for result in prediction:
        if result == 1:
            chris = chris + 1
        elif result == 0:
            sarah = sarah + 1

    print('Chris %d' % chris)
    print('Sarah %d' % sarah)

#train_and_predict(10)
#train_and_predict(100)
#train_and_predict(1000)
train_and_predict(10000)
