# Set working directory
import os
os.chdir('H:\Machine Learning\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data_Preprocessing')
print(os.getcwd())


# Data PreProcessing

# Step 1: Setup
# Importing the libraries
import numpy as np
np.set_printoptions(threshold = np.inf)
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
dataset

# Create feature and target variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Step 2: Take care of missing data
from sklearn.preprocessing import Imputer # Import library
imputer = Imputer(missing_values = 'NaN', strategy = 'mean',axis = 0) # Create imputer object
imputer = imputer.fit(x[:,1:3]) 
x[:,1:3]= imputer.transform(x[:,1:3])

# Step 3: Encode Categorical Variables
# Machine learning models are based on mathematical equations.
# Keeping the categorical variables would cause problems because we only want numbers in the equation
# That is why we need to encode the categorical variables into numbers

# Encode categorical variables of x matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Import library
labelencoder_x = LabelEncoder() # Create object of label encoder class
x[:,0]=labelencoder_x.fit_transform(x[:,0]) # Apply label encoder object

# Since there are multiple categories, need to let machine learning model no there is no order to them
onehotencoder =OneHotEncoder(categorical_features = [0]) # Create OneHotEncoder object
x = onehotencoder.fit_transform(x).toarray() #  Fit object ot matrix x

# Encode categorical variable of y matrix
# Since there is only one category, only need to do label coder steps
labelencoder_y = LabelEncoder() # Create object of label encoder class
y = labelencoder_y.fit_transform(y) # Apply label encoder object

# STEP 4: Split dataset in training and test sets

# import library
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# STEP 5: Feature Scaling
#  Alot of machine learning models are based on Euclidean distance between two data points (square root of the sum of the squared coordinates)
# Think of age as x coordinate and salary as y coordinate
# Many ways to scale data, common way is standardization (for each observation of each feature subtract mean and divide by standard deviation) or Normalization (subtract observation feature x by the min value of all the feature value x and divide by difference of max and min feature values)

# Import library
from sklearn.preprocessing import StandardScaler

# Create object of this class
sc_x = StandardScaler()

# Fit and transform x training set
x_train = sc_x.fit_transform(x_train)

# Transform x test set (don't need to fit because it is already fitted to training set)
x_test = sc_x.transform(x_test)

# For regression, when the dependent variable could take on a huge range of values, would need to apply feature scaling to the depedent variable y as well






