#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.stem.porter import PorterStemmer


def decisionTree(inputs, scores):
    df = pd.DataFrame({'tweet': inputs,
                       'score': scores})

    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df.tweet).toarray()
    y = df.score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=324)

    classifier = DecisionTreeClassifier(max_leaf_nodes=600, random_state=0)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_train, y_train))

    y_predicted = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_predicted))
    print('Accuracy: ', accuracy_score(y_test, y_predicted))
    print('Precision: ', precision_score(y_test, y_predicted, average='weighted', zero_division=1))
    print('Recall: ', recall_score(y_test, y_predicted, average='weighted', zero_division=1))
    print('F1: ', f1_score(y_test, y_predicted, average='weighted'))


def decisionTreeAsking(inputs, scores):
    df = pd.DataFrame({'tweet': inputs,
                       'score': scores})

    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df.tweet).toarray()
    y = df.score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=324)

    classifier = DecisionTreeClassifier(max_leaf_nodes=600, random_state=0)
    classifier.fit(X_train, y_train)

    y_predicted = classifier.predict(X_test)

    ps = PorterStemmer()
    tweet = input("Tweet: ")
    tweet = re.sub('[^a-zA-Z]', ' ', tweet).split()
    tweet = ' '.join([ps.stem(w) for w in tweet])
    X = vectorizer.transform([tweet]).toarray()

    result = classifier.predict(X)

    print("From 0 to 1, 0 being 'non-ironic' and 1 'ironic', you tweet scored ", result)
    


# In[ ]:




