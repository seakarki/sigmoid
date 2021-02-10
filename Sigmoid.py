#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


# In[2]:


# Now load in the dataset with Pandas
dataset = pd.read_csv('./Downloads/heart.csv')
# Print out part of the dataset to ensure proper loading
df = dataset.copy()
print(df.head(5))


# In[3]:


#Lets Separate dependent and independent variable
x= df[['Treatment of Anger', 'Trait Anxiety']]
y = df[['2nd Heart Attack']]
x.head()


# In[4]:


y.head()


# In[5]:


# lets split the data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=31)
# Create an instance of the scaler and apply it to the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[32]:


# Display the test data sets 
X_test


# In[33]:


# import the logisticregression algorithm to fit to the training data
from sklearn.linear_model import LogisticRegression
logistic_clf = LogisticRegression(random_state = 0)
logistic_clf.fit(X_train, y_train)


# In[34]:


## Lets print out the Coefficient and intercept which we find from the logistic algorithm
print(logistic_clf.coef_)
print(logistic_clf.intercept_)


# In[42]:


X_test


# In[44]:


# we do have two coefficent for two independent variable
𝑊1= -0.39016116 # Coefficent Treatment of Anger
𝑊2= 0.93324057 # Coefficent Trait Anxiety
𝑊0= -0.15094771 # intercept or bias


# In[45]:


#(Equation) z:= 𝑊1*Person + 𝑊2*Treatment of Anger + W3*Trait Anxiety + 𝑊0
z1 =((1.19522861*𝑊1) +(-0.47623957*𝑊2) +𝑊0)
z1


# In[46]:


# Apply an above equation to get the real value as value
z2 =((X_test[1][0]*𝑊1) + (X_test[1][1]*𝑊2)  +𝑊0)
z3 =((X_test[2][0]*𝑊1) + (X_test[2][1]*𝑊2)  +𝑊0)
z1,z2,z3


# In[47]:


# we obtain the same real value using the decison function
logistic_clf.decision_function(X_test)


# In[19]:


# Let's define our sigmoid function 
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# In[20]:


# lets convert the real value between 0 and 1 using the sigmoid function
sigmoid(z1),sigmoid(z2),sigmoid(z3)


# In[21]:


# Below we use predict_proba to obtain the same as above to convert between 0 and 1 
logistic_clf.predict_proba(X_test)[:,1]


# In[22]:


#  Since we obtain the probablity between 0 and 1 from the simoid function 
# we will use Decison boundary  or threshold to classify the class
y_pred = logistic_clf.predict(X_test)
y_pred


# In[ ]:









