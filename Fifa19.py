#!/usr/bin/env python
# coding: utf-8

# In[1]:


#checking the directory
import os
print(os.listdir())


# In[2]:


# importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import gc


# In[3]:


#to get the graphs inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataSet = pd.read_csv('data.csv')


# In[5]:


dataSet.head(10)


# In[6]:


dataSet['ValueK'] = dataSet['Value'].str.replace('€','').str.replace('M','000').str.replace('K','')
dataSet['WageK'] = dataSet['Wage'].str.replace('€','').str.replace('K','')
dataSet['ReleaseClauseK'] = dataSet['Release Clause'].str.replace('€','').str.replace('M','000').str.replace('K','')
dataSet.drop(['Value','Wage','Release Clause'],axis=1,inplace=True)


# In[7]:


#converting string into int
dataSet['ValueK'] = pd.to_numeric(dataSet['ValueK'])
dataSet['WageK'] = pd.to_numeric(dataSet['WageK'])
dataSet['ReleaseClauseK'] = pd.to_numeric(dataSet['ReleaseClauseK'])


# In[8]:


# Splitting into Dependent and independent matrix
X = dataSet['WageK'].values
y= dataSet['ValueK'].values


# In[9]:


#Reshaping the matrix into 2D matrix
X = X.reshape(-1, 1)


# In[10]:


#convert the dataSet into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)


# In[11]:


#passing dataSet into Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[12]:


regressor.fit(X_train, y_train)


# In[13]:


#create a predictor
y_predictor = regressor.predict(X_test)


# In[14]:


y_predictor


# # Plotting the Graph with Training DataSet

# In[15]:


fig = plt.figure(figsize=(10,10))
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('How different is Market value from Wages of a player in € ')
plt.xlabel('Wages (in Thousands)')
plt.ylabel('Market value (in Millions)')


# # Plotting the dataSet with test dataSet

# In[16]:


fig = plt.figure(figsize=(10,10))
plt.scatter(X_test, y_test)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('How different is Market value from Wages of a player in € ')
plt.xlabel('Wages (in Thousands)')
plt.ylabel('Market value (in Millions)')


# In[17]:


preferredFoot = dataSet.groupby('Preferred Foot')['ID'].count()


# In[18]:


preferredFoot = preferredFoot/preferredFoot.sum()*100


# In[19]:


fig = plt.figure(figsize=(7,7))
plt.bar(x = ['Left', 'Right'], height = preferredFoot.values)
plt.title('Preferred Foot (in %)')


# In[20]:


# getting the details of the dataSet
datainfo = dataSet[['Position', 'Preferred Foot']].groupby('Position')['Preferred Foot'].value_counts().unstack()


# In[21]:


#Plotting the graph
datainfo['Left']= datainfo['Left']/datainfo['Left'].sum()
datainfo['Right']= datainfo['Right']/datainfo['Right'].sum()
fig, ax = plt.subplots(figsize=(15,7));
datainfo['Right'].plot(ax=ax);
datainfo['Left'].plot(ax=ax);
plt.legend(['Right','Left'])
plt.title("Position vs Foot");


# ## Can we predict the market value of a player based on their attributes ?

# In[22]:


#following CRISP DM methodology to answer this question 
#features chosen
datavalue = dataSet[['Preferred Foot','Position','Crossing', 'Finishing', 'HeadingAccuracy',
       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
       'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
       'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision',
       'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
       'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes','ValueK']]

#Several Attributes, Positioning and Prefered Foot are chosen as features to predict the Market Value


# In[23]:


# Get one hot encoding of column - position and preferred foot
one_hot = pd.get_dummies(datavalue[['Position','Preferred Foot']])
one_hot

# Drop columns as it is now encoded
datavalue = datavalue.drop(columns = ['Position','Preferred Foot'],axis = 1)

# Join the encoded df
datavalue = datavalue.join(one_hot)
datavalue.head(10)


# ##### One hot encoding is chosen above, because we want to retain the value of each column and all values are to be treated equally, it seems appropriate over other methods like 'label encoding' which gives different ranking to values

# In[24]:


(datavalue.isnull().sum()/datavalue.count())*100


# In[25]:


datavalue.dropna(inplace= True)
datavalue.isnull().sum() #no null left


# In[26]:


#plt.figure(figsize=(20,8))
plt.figure(figsize=(15,5))
sns.countplot(x = 'Position',
              data = dataSet,
              order = dataSet['Position'].value_counts().index,palette=sns.color_palette("Blues_d",n_colors=27));


# ### Strikers Goalkeepers and Centre backs are the top three positions.

# # Top 11 Clubs with the highest median wages 

# In[27]:


dataSet[['WageK','Club']].groupby(['Club'])['WageK'].median().sort_values(ascending=False).head(11)


# # Top 11 Players with the highest Release Clause 

# In[28]:


dataSet[['ReleaseClauseK','Name']].sort_values(by='ReleaseClauseK',ascending=False)['Name'].head(11).reset_index(drop=True)


# In[ ]:




