
# coding: utf-8

# Using XGBoost is easy. Maybe too easy, considering it's generally considered the best ML algorithm around right now.
# 
# To install it, just:
# 
# pip install xgboost
# 
# Let's experiment using the Iris data set. This data set includes the width and length of the petals and sepals of many Iris flowers, and the specific species of Iris the flower belongs to. Our challenge is to predict the species of a flower sample just based on the sizes of its petals. We'll revisit this data set later when we talk about principal component analysis too.

# In[1]:


from sklearn.datasets import load_iris

iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))


# Let's divide our data into 20% reserved for testing our model, and the remaining 80% to train it with. By withholding our test data, we can make sure we're evaluating its results based on new flowers it hasn't seen before. Typically we refer to our features (in this case, the petal sizes) as X, and the labels (in this case, the species) as y.

# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)


# Now we'll load up XGBoost, and convert our data into the DMatrix format it expects. One for the training data, and one for the test data.

# In[3]:


import xgboost as xgb

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)


# Now we'll define our hyperparameters. We're choosing softmax since this is a multiple classification problem, but the other parameters should ideally be tuned through experimentation.

# In[4]:


param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 3} 
epochs = 10 


# Let's go ahead and train our model using these parameters as a first guess.

# In[5]:


model = xgb.train(param, train, epochs)


# Now we'll use the trained model to predict classifications for the data we set aside for testing. Each classification number we get back corresponds to a specific species of Iris.

# In[6]:


predictions = model.predict(test)


# In[7]:


print(predictions)


# Let's measure the accuracy on the test data...

# In[8]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)


# Holy crow! It's perfect, and that's just with us guessing as to the best hyperparameters!
# 
# Normally I'd have you experiment to find better hyperparameters as an activity, but you can't improve on those results. Instead, see what it takes to make the results worse! How few epochs (iterations) can I get away with? How low can I set the max_depth? Basically try to optimize the simplicity and performance of the model, now that you already have perfect accuracy.
