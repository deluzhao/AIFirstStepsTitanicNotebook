#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Titanic Data and Data Transformation

# In[46]:


file = pd.read_csv("/Users/deluzhao/Downloads/titanic.csv")
file['age'] = file.fillna(file['age'].sum()/file['age'].count())
file['fare'] = file.fillna(file['fare'].sum()/file['fare'].count())
file = file.replace(to_replace = 'male', value = 1)
file = file.replace(to_replace = 'female', value = 0)
X = file.drop(labels = ['survived', 'name', 'ticket', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)
y = file['survived']
from sklearn.model_selection import train_test_split
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.20)


# In[47]:


file.isnull().sum()


# In[48]:


file.head()


# # Using Decision Tree 

# In[49]:


from sklearn.tree import DecisionTreeClassifier
b = DecisionTreeClassifier(max_depth=5)
b.fit(trainx, trainy)


# In[50]:


predicty = b.predict(testx)


# # Checking Accuracy

# In[51]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testy, predicty))
print(classification_report(testy, predicty))


# # Visualizing Tree

# In[52]:


plt.figure(figsize=(50,10))
dot_data = sklearn.tree.plot_tree(b, feature_names=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'], class_names=['0','1'], filled=True, rounded=True, fontsize=14)


# In[53]:


importance = b.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d Score: %.5f' % (i,v), )
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[ ]:




