#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd


# In[19]:


import numpy as np


# In[20]:


dt=pd.read_csv("Spam.csv")
dt.head(10)


# In[21]:


dt['spam']=dt['type'].map( {'spam':1,'ham':0} ).astype(int)
dt.head(5)


# In[22]:


print("Columns in the given data:")
for col in dt.columns:
    print(col)


# In[23]:


t=len(dt['type'])
print("No. of rows in review column:",t)
t=len(dt['text'])
print("No. of rows in liked column:",t)


# In[24]:


dt['text'][1]


# In[25]:


def tokenizer(text):
    return text.split()


# In[26]:


dt['text']=dt['text'].apply(tokenizer)


# In[27]:


dt['text'][1]


# In[28]:


dt['text'][1]


# In[29]:


from nltk.stem.snowball import SnowballStemmer
porter=SnowballStemmer("english",ignore_stopwords=False)


# In[30]:


def stem_it(text):
    return [porter.stem(word) for word in text]


# In[31]:


dt['text']=dt['text'].apply(stem_it)


# In[32]:


dt['text'][1]


# In[34]:


dt['text'][112]


# In[38]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[39]:


def lemmit_it(text):
    return[lemmatizer.lemmatize(word,pos="a") for word in text]


# In[45]:


dt['text']=dt['text'].apply(lemmit_it)


# In[46]:


dt['text'][112]


# In[50]:


dt['text'][65]


# In[52]:


import nltk
nltk.download('stopwords')


# In[58]:


from nltk.corpus import stopwords
stop_words=stopwords.words('english')


# In[59]:


def stop_it(text):
    review=[word for word in text if not word in stop_words]
    return review


# In[60]:


dt['text']=dt['text'].apply(stop_it)


# In[61]:


dt['text'][65]


# In[62]:


dt.head(10)


# In[63]:


dt['text']=dt['text'].apply(' '.join)


# In[64]:


dt.head()


# In[65]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
y=dt.spam.values
x=tfidf.fit_transform(dt['text'])


# In[66]:


from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,random_state=1,test_size=0.2,shuffle=False)


# In[68]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_text)
from sklearn.metrics import accuracy_score
acc_log = accuracy_score(y_pred, y_text)*100
print("Accuracy:",acc_log )


# In[70]:


from sklearn.svm import LinearSVC
linear_svc = LinearSVC(random_state=0)
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_text)
acc_linear_svc =accuracy_score(y_pred, y_text) * 100
print("Accuracy:",acc_linear_svc)


# In[ ]:




