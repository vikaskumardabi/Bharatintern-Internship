#!/usr/bin/env python
# coding: utf-8

# # Bharat Intern
# Business Analytics Intern
# 
# Task 1 : Performance Rating and Attrition
# 
# Name : Vikas Kumar Dabi
# 
# To begin, I need to install the necessary libraries and load the dataset .

# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Display the first few rows of the dataset
print(data.head())


# Next, I analyze the relationship between attrition and different factors

# 1)Distance from home to office :

# In[14]:


# Boxplot showing the relationship between attrition and distance from home
sns.boxplot(x='Attrition', y='DistanceFromHome', data=data)
plt.title('Attrition vs Distance from Home')
plt.show()


# 2)Job role impact on attrition:

# In[15]:


# Countplot showing the relationship between attrition and job role
sns.countplot(x='JobRole', hue='Attrition', data=data)
plt.title('Attrition by Job Role')
plt.xticks(rotation=90)
plt.show()


# 3)Performance rating and attrition:

# In[16]:


# Boxplot showing the relationship between attrition and performance rating
sns.boxplot(x='Attrition', y='PerformanceRating', data=data)
plt.title('Attrition vs Performance Rating')
plt.show()


# # THE END
