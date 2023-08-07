#!/usr/bin/env python
# coding: utf-8

# # Bharat Intern
# Business Analytics Intern
# 
# Task 2 : Success of an Upcoming Movie
# 
# Name : Vikas Kumar Dabi
# 
# To begin, I need to install the necessary libraries and load the dataset.

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


df = pd.read_csv('movie_success_rate.csv')


# In[10]:


df.shape


# In[11]:


df.head()


# In[16]:


df.columns


# In[17]:


df['Genre'].value_counts()


# In[18]:


df['Director'].value_counts()


# In[19]:


df['Actors'].value_counts()


# In[20]:


import seaborn as sns
sns.heatmap(df.isnull())
     


# In[21]:


df = df.fillna(df.median())


# In[22]:


df.columns


# In[23]:


x = df[['Year',
       'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)',
       'Metascore', 'Action', 'Adventure', 'Aniimation', 'Biography', 'Comedy',
       'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War',
       'Western']]
y = df['Success']


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1,stratify=y)


# In[25]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)


# In[26]:


log.score(x_test,y_test)


# In[27]:


from sklearn.metrics import confusion_matrix
clf = confusion_matrix(y_test,log.predict(x_test))


# In[28]:


sns.heatmap(clf,annot=True)


# In[29]:


#normalising all columns
x_train_opt = x_train.copy()
x_test_opt = x_test.copy()


# In[30]:


from sklearn.preprocessing import StandardScaler
x_train_opt = StandardScaler().fit_transform(x_train_opt)
x_test_opt = StandardScaler().fit_transform(x_test_opt)


# In[31]:


log.fit(x_train_opt,y_train)


# In[32]:


log.score(x_test_opt,y_test)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=40)
kn.fit(x_train,y_train)


# In[34]:


kn.score(x_test,y_test)


# In[35]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree.score(x_test,y_test)


# In[36]:


tree.score(x_train,y_train)


# In[ ]:




