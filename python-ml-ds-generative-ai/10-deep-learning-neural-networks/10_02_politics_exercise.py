
# coding: utf-8

# # Keras Exercise
# 
# ## Predict political party based on votes
# 
# As a fun little example, we'll use a public data set of how US congressmen voted on 17 different issues in the year 1984. Let's see if we can figure out their political party based on their votes alone, using a deep neural network!
# 
# For those outside the United States, our two main political parties are "Democrat" and "Republican." In modern times they represent progressive and conservative ideologies, respectively.
# 
# Politics in 1984 weren't quite as polarized as they are today, but you should still be able to get over 90% accuracy without much trouble.
# 
# Since the point of this exercise is implementing neural networks in Keras, I'll help you to load and prepare the data.
# 
# Let's start by importing the raw CSV file using Pandas, and make a DataFrame out of it with nice column labels:

# In[1]:


import pandas as pd

feature_names =  ['party','handicapped-infants', 'water-project-cost-sharing', 
                    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                    'el-salvador-aid', 'religious-groups-in-schools',
                    'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                    'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                    'education-spending', 'superfund-right-to-sue', 'crime',
                    'duty-free-exports', 'export-administration-act-south-africa']

voting_data = pd.read_csv('house-votes-84.data.txt', na_values=['?'], 
                          names = feature_names)
voting_data.head()


# We can use describe() to get a feel of how the data looks in aggregate:

# In[2]:


voting_data.describe()


# We can see there's some missing data to deal with here; some politicians abstained on some votes, or just weren't present when the vote was taken. We will just drop the rows with missing data to keep it simple, but in practice you'd want to first make sure that doing so didn't introduce any sort of bias into your analysis (if one party abstains more than another, that could be problematic for example.)

# In[3]:


voting_data.dropna(inplace=True)
voting_data.describe()


# Our neural network needs normalized numbers, not strings, to work. So let's replace all the y's and n's with 1's and 0's, and represent the parties as 1's and 0's as well.

# In[4]:


voting_data.replace(('y', 'n'), (1, 0), inplace=True)
voting_data.replace(('democrat', 'republican'), (1, 0), inplace=True)


# In[5]:


voting_data.head()


# Finally let's extract the features and labels in the form that Keras will expect:

# In[6]:


all_features = voting_data[feature_names].drop('party', axis=1).values
all_classes = voting_data['party'].values


# OK, so have a go at it! You'll want to refer back to the slide on using Keras with binary classification - there are only two parties, so this is a binary problem. This also saves us the hassle of representing classes with "one-hot" format like we had to do with MNIST; our output is just a single 0 or 1 value.
# 
# Also refer to the scikit_learn integration slide, and use cross_val_score to evaluate your resulting model with 10-fold cross-validation.
# 
# **If you're using tensorflow-gpu on a Windows machine** by the way, you probably *do* want to peek a little bit at my solution - if you run into memory allocation errors, there's a workaround there you can use.
# 
# Try out your code here; be sure to have scikeras installed if you don't already (you may need to launch this notebook with admin privleges, or just install it from your Anaconda prompt):

# In[ ]:


get_ipython().system('pip install scikeras')


# ## My implementation is below
# 
# # No peeking!
# 
# ![title](peek.jpg)

# In[8]:


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

def create_model():
    model = Sequential([
        Dense(32, input_dim=16, kernel_initializer='normal', activation='relu'),
        Dense(16, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap our Keras model with SciKeras KerasClassifier
estimator = KerasClassifier(model=create_model, epochs=100, verbose=0)

# Assuming all_features and all_classes are defined and properly preprocessed
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
print(cv_scores.mean())


# 94% without even trying too hard! Did you do better? Maybe more neurons, more layers, or Dropout layers would help even more.
