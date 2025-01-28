
# coding: utf-8

# # Percentiles

# In[1]:


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

vals = np.random.normal(0, 0.5, 10000)

plt.hist(vals, 50)
plt.show()


# In[2]:


np.percentile(vals, 50)


# In[3]:


np.percentile(vals, 90)


# In[4]:


np.percentile(vals, 20)


# ## Activity

# Experiment with different parameters when creating the test data. What effect does it have on the percentiles?
