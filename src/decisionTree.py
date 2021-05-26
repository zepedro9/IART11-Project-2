#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def proccess(trainData, trainScores, testData, testScores, task, trainDataProcessTime):
    start = time.time()
    model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())
    model.fit(trainData, trainScores)
    predictions = model.predict(testData)
    end = time.time()
    
    if(task == "A"):
        labels=["0", "1"]
        print("------------ TASK A -------------")
        print("-------- DECISION TREES --------")
        print("Ironic = 0 | Non-ironic = 1")
    else:
        labels=["0", "1", "2", "3"]
        print("------------ TASK B -------------")
        print("-------- DECISION TREES --------")
        print("\n")
        print("Ironic with polarity contrast = 0 | Ironic without polarity contrast = 1 | Situationaly ironic = 2 | Non-ironic = 3")
    
    mat = confusion_matrix(testScores, predictions)
    sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=labels,yticklabels=labels)
    plt.xlabel("Actual irony")
    plt.ylabel("Predicted irony")
    plt.show()
    print("The accuracy is {}".format(accuracy_score(testScores, predictions)))
    print("The precision is {}".format(precision_score(testScores, predictions, average='weighted')))
    print("The recall is {}".format(recall_score(testScores, predictions, average='weighted')))
    print("The f1 is {}".format(f1_score(testScores, predictions, average='weighted')))
    testDataProcessTime = end - start
    print("\n")
    print("Time taken to train model: ", trainDataProcessTime, "seconds")
    print("Time taken to test model: ", testDataProcessTime, "seconds")


# In[ ]:




