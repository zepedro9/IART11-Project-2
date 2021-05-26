#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def clean_tweet(old):
    new = remove_identifications(old)
    new = remove_links(new)
    new = convert_lower_case(new)
    new = remove_punctuation(new)
    new = remove_apostrophe(new)
    new = remove_single_characters(new)
    new = remove_stop_words(new)
    new = stemming(new)
    #new = lemmatization(new)
    new = remove_punctuation(new)
    return str(old)

def remove_identifications(old):
    new = " ".join(filter(lambda x: x[0] != '@', old.split()))
    return new

def remove_links(old):
    return re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", old)

def convert_lower_case(old):
    return np.char.lower(old)

def remove_punctuation(old):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        old = np.char.replace(old, i, ' ')
    return old

def remove_apostrophe(old):
    return np.char.replace(old, "'", "")

def remove_single_characters(old):
    new = ""
    tmp = str(old).split()
    for w in tmp:
        if len(w) > 1:
            new = new + " " + w
    return new

def remove_stop_words(old):
    stop_words = stopwords.words('english')
    new = ""
    for word in old.split():
        if word not in stop_words:
            new = new + " " + word
    return new

def stemming(old):
    ps = PorterStemmer()
    new = re.sub('[^a-zA-Z]', ' ', old)
    return ps.stem(new)

def lemmatization(old):
    lm = WordNetLemmatizer()
    new = re.sub('[^a-zA-Z]', ' ', old)
    return lm.lemmatize(new)


# In[ ]:





# In[ ]:





# In[ ]:




