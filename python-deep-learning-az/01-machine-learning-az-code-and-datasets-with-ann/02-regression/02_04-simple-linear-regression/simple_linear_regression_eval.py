# Salary Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv') # change name
X = dataset.iloc[:, :-1].values # create matrix of features
y = dataset.iloc[:, 1].values # create dependent variable vector. change independent variable index

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # Change test size

# Feature Scaling not needed because the library used to build simple linear regression model will take care of that
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training Set

# Import library
from sklearn.linear_model import LinearRegression # From linear_model library import LinearRegression class

# Create object of Linear Regression class by calling it
regressor = LinearRegression()

# Fit regressor object to training set by using the .fit method of the LinearRegression class
# Fitting Simple Linear Regression to the Training set
# We created a machine (simple linear regression model) and made it learn correlations on the training set so that the machine can predict salary based on its learning experience.
model = regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test) # vector of predictions of dependent variable

# Visualizing the Training set results
plt.scatter(X_train, y_train,color = 'red') # plot real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # plot regression line of predicted values on training set
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red') # plot real values of test set
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # plot predicted values of test set # Same regression line as above
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Evaluate results
from sklearn import metrics

# Calculate metrics
print('MAE',metrics.mean_absolute_error(y_test, y_pred)) #MAE is the easiest to understand, because it's the average error
print('MSE',metrics.mean_squared_error(y_test, y_pred)) #MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units (target units)

# Calculated R Squared
print('R^2 =',metrics.explained_variance_score(y_test,y_pred))

