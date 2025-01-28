
# coding: utf-8

# # Multiple Regression

# Let's grab a small little data set of Blue Book car values:

# In[1]:


import pandas as pd

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')


# In[2]:


get_ipython().magic('matplotlib inline')
import numpy as np
df1=df[['Mileage','Price']]
bins =  np.arange(0,50000,10000)
groups = df1.groupby(pd.cut(df1['Mileage'],bins)).mean()
print(groups.head())
groups['Price'].plot.line()


# We can use pandas to split up this matrix into the feature vectors we're interested in, and the value we're trying to predict.
# 
# Note how we are avoiding the make and model; regressions don't work well with ordinal values, unless you can convert them into some numerical order that makes sense somehow.
# 
# Let's scale our feature data into the same range so we can easily compare the coefficients we end up with.

# In[3]:


import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].values)

# Add a constant column to our model so we can have a Y-intercept
X = sm.add_constant(X)

print (X)

est = sm.OLS(y, X).fit()

print(est.summary())


# The table of coefficients above gives us the values to plug into an equation of form:
#     B0 + B1 * Mileage + B2 * cylinders + B3 * doors
#     
# In this example, it's pretty clear that the number of cylinders is more important than anything based on the coefficients.
# 
# Could we have figured that out earlier?

# In[4]:


y.groupby(df.Doors).mean()


# Surprisingly, more doors does not mean a higher price! (Maybe it implies a sport car in some cases?) So it's not surprising that it's pretty useless as a predictor here. This is a very small data set however, so we can't really read much meaning into it.
# 
# How would you use this to make an actual prediction? Start by scaling your multiple feature variables into the same scale used to train the model, then just call est.predict() on the scaled features:

# In[5]:


scaled = scale.transform([[45000, 8, 4]])
scaled = np.insert(scaled[0], 0, 1) #Need to add that constant column in again.
print(scaled)
predicted = est.predict(scaled)
print(predicted)


# ## Activity

# Mess around with the fake input data, and see if you can create a measurable influence of number of doors on price. Have some fun with it - why stop at 4 doors?
