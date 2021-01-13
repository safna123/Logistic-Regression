#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd


# In[58]:


import numpy as np


# In[59]:


import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


df=pd.read_csv('titanic.csv')


# In[61]:


df.shape


# In[62]:


df.head()


# In[63]:


df=pd.get_dummies(df)


# In[64]:


df.fillna(0,inplace=True)


# In[65]:


df.shape


# In[66]:


df.head()


# In[67]:


train=df[0:600]


# In[68]:


test=df[600:]


# In[69]:


x_train=train.drop('Survived',axis=1)


# In[70]:


y_train=train['Survived']


# In[71]:


x_test=test.drop('Survived',axis=1)


# In[72]:


test_p=test['Survived']


# In[73]:


from sklearn.linear_model import LogisticRegression


# In[74]:


loreg=LogisticRegression()


# In[75]:


#cost function and prediction


# In[76]:


loreg.fit(x_train,y_train)


# In[77]:


pred=loreg.predict(x_test)


# In[78]:


print(pred)


# In[79]:


#score(accuracy)


# In[80]:


loreg.score(x_test,test_p)


# In[81]:


loreg.score(x_train,y_train)


# In[ ]:





# In[ ]:




