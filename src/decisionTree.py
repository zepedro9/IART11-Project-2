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

def proccess(trainData, trainScores, testData, testScores, task):
    start = time.time()
    model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())
    model.fit(trainData, trainScores)
    end = time.time()
    trainDataProcessTime = end - start
    start = time.time()
    predictions = model.predict(testData)
    end = time.time()
    testDataProcessTime = end - start
    
    if(task == "A"):
        labels=["0", "1"]
        print("------------ TASK A -------------")
        print("-------- DECISION TREES --------")
        print("Non-ironic = 0 | Ironic = 1")
    else:
        labels=["0", "1", "2", "3"]
        print("------------ TASK B -------------")
        print("-------- DECISION TREES --------")
        print("\n")
        print("Non-ironic = 0 | Situationaly ironic = 1 | Ironic without polarity contrast = 2 | Ironic with polarity contrast = 3 ")
    
    mat = confusion_matrix(testScores, predictions)
    sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=labels,yticklabels=labels)
    plt.xlabel("Actual irony")
    plt.ylabel("Predicted irony")
    plt.show()
    print("The accuracy is {value:.5f}%".format(value = accuracy_score(testScores, predictions)))
    print("The precision is {value:.5f}%".format(value = precision_score(testScores, predictions, average='weighted')))
    print("The recall is {value:.5f}%".format(value = recall_score(testScores, predictions, average='weighted')))
    print("The f1 is {value:.5f}%".format(value = f1_score(testScores, predictions, average='weighted')))
    print("\n")
    print("Time taken to train model: {value:.5f} seconds".format(value = trainDataProcessTime))
    print("Time taken to test model: {value:.5f} seconds".format(value = testDataProcessTime))


# In[ ]:




