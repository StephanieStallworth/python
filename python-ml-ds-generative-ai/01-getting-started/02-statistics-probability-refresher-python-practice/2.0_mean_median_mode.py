
# coding: utf-8

# # Mean, Median, Mode, and introducing NumPy

# ## Mean vs. Median

# Let's create some fake income data, centered around 27,000 with a normal distribution and standard deviation of 15,000, with 10,000 data points. (We'll discuss those terms more later, if you're not familiar with them.)
# 
# Then, compute the mean (average) - it should be close to 27,000:

# In[1]:


import numpy as np

incomes = np.random.normal(27000, 15000, 10000)
np.mean(incomes)


# We can segment the income data into 50 buckets, and plot it as a histogram:

# In[2]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.hist(incomes, 50)
plt.show()


# Now compute the median - since we have a nice, even distribution it too should be close to 27,000:

# In[3]:


np.median(incomes)


# Now we'll add Jeff Bezos into the mix. Darn income inequality!

# In[4]:


incomes = np.append(incomes, [1000000000])


# The median won't change much, but the mean does:

# In[5]:


np.median(incomes)


# In[6]:


np.mean(incomes)


# ## Mode

# Next, let's generate some fake age data for 500 people:

# In[9]:


ages = np.random.randint(18, high=90, size=500)
ages


# In[10]:


from scipy import stats
stats.mode(ages)

