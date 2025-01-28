
# coding: utf-8

# # Final Project
# 
# ## Predict whether a mammogram mass is benign or malignant
# 
# We'll be using the "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
# 
# This data contains 961 instances of masses detected in mammograms, and contains the following attributes:
# 
# 
#    1. BI-RADS assessment: 1 to 5 (ordinal)  
#    2. Age: patient's age in years (integer)
#    3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
#    4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
#    5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
#    6. Severity: benign=0 or malignant=1 (binominal)
#    
# BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute and so we will discard it. The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.
# 
# Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with well, they are close enough to ordinal that we shouldn't just discard them. The "shape" for example is ordered increasingly from round to irregular.
# 
# A lot of unnecessary anguish and surgery arises from false positives arising from mammogram results. If we can build a better way to interpret them through supervised machine learning, it could improve a lot of lives.
# 
# ## Your assignment
# 
# Apply several different supervised machine learning techniques to this data set, and see which one yields the highest accuracy as measured with K-Fold cross validation (K=10). Apply:
# 
# * Decision tree
# * Random forest
# * KNN
# * Naive Bayes
# * SVM
# * Logistic Regression
# * And, as a bonus challenge, a neural network using Keras.
# 
# The data needs to be cleaned; many rows contain missing data, and there may be erroneous data identifiable as outliers as well.
# 
# Remember some techniques such as SVM also require the input data to be normalized first.
# 
# Many techniques also have "hyperparameters" that need to be tuned. Once you identify a promising approach, see if you can make it even better by tuning its hyperparameters.
# 
# I was able to achieve over 80% accuracy - can you beat that?
# 
# Below I've set up an outline of a notebook for this project, with some guidance and hints. If you're up for a real challenge, try doing this project from scratch in a new, clean notebook!
# 

# ## Let's begin: prepare your data
# 
# Start by importing the mammographic_masses.data.txt file into a Pandas dataframe (hint: use read_csv) and take a look at it.

# Make sure you use the optional parmaters in read_csv to convert missing data (indicated by a ?) into NaN, and to add the appropriate column names (BI_RADS, age, shape, margin, density, and severity):

# Evaluate whether the data needs cleaning; your model is only as good as the data it's given. Hint: use describe() on the dataframe.

# There are quite a few missing values in the data set. Before we just drop every row that's missing data, let's make sure we don't bias our data in doing so. Does there appear to be any sort of correlation to what sort of data has missing fields? If there were, we'd have to try and go back and fill that data in.

# If the missing data seems randomly distributed, go ahead and drop rows with missing data. Hint: use dropna().

# Next you'll need to convert the Pandas dataframes into numpy arrays that can be used by scikit_learn. Create an array that extracts only the feature data we want to work with (age, shape, margin, and density) and another array that contains the classes (severity). You'll also need an array of the feature name labels.

# Some of our models require the input data to be normalized, so go ahead and normalize the attribute data. Hint: use preprocessing.StandardScaler().

# ## Decision Trees
# 
# Before moving to K-Fold cross validation and random forests, start by creating a single train/test split of our data. Set aside 75% for training, and 25% for testing.

# Now create a DecisionTreeClassifier and fit it to your training data.

# Display the resulting decision tree.

# Measure the accuracy of the resulting decision tree model using your test data.

# Now instead of a single train/test split, use K-Fold cross validation to get a better measure of your model's accuracy (K=10). Hint: use model_selection.cross_val_score

# Now try a RandomForestClassifier instead. Does it perform better?

# ## SVM
# 
# Next try using svm.SVC with a linear kernel. How does it compare to the decision tree?

# ## KNN
# How about K-Nearest-Neighbors? Hint: use neighbors.KNeighborsClassifier - it's a lot easier than implementing KNN from scratch like we did earlier in the course. Start with a K of 10. K is an example of a hyperparameter - a parameter on the model itself which may need to be tuned for best results on your particular data set.

# Choosing K is tricky, so we can't discard KNN until we've tried different values of K. Write a for loop to run KNN with K values ranging from 1 to 50 and see if K makes a substantial difference. Make a note of the best performance you could get out of KNN.

# ## Naive Bayes
# 
# Now try naive_bayes.MultinomialNB. How does its accuracy stack up? Hint: you'll need to use MinMaxScaler to get the features in the range MultinomialNB requires.

# ## Revisiting SVM
# 
# svm.SVC may perform differently with different kernels. The choice of kernel is an example of a "hyperparamter." Try the rbf, sigmoid, and poly kernels and see what the best-performing kernel is. Do we have a new winner?

# ## Logistic Regression
# 
# We've tried all these fancy techniques, but fundamentally this is just a binary classification problem. Try Logisitic Regression, which is a simple way to tackling this sort of thing.

# ## Neural Networks
# 
# As a bonus challenge, let's see if an artificial neural network can do even better. You can use Keras to set up a neural network with 1 binary output neuron and see how it performs. Don't be afraid to run a large number of epochs to train the model if necessary.

# ## Do we have a winner?
# 
# Which model, and which choice of hyperparameters, performed the best? Feel free to share your results!
