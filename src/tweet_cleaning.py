#!/usr/bin/env python
# coding: utf-8

# In[26]:


import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def clean_tweet(old):
    new = remove_identifications(old)
    new = remove_links(new)
    new = stemming(new)
    return new.lower()

def remove_identifications(old):
    new = " ".join(filter(lambda x: x[0] != '@', old.split()))
    return new

def remove_links(old):
    return re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", old)

def stemming(old):
    ps = PorterStemmer()
    new = re.sub('[^a-zA-Z]', ' ', old)
    new = new.split()
    new = ' '.join([ps.stem(w) for w in new if not w in set(stopwords.words('english'))])
    return new


# In[ ]:





# In[ ]:





# In[ ]:




