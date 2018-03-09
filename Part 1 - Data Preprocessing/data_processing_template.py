# Data Preprocessing

# Importing the libraries
import numpy as np # multi-dimensional arrays and matrices
import matplotlib.pyplot as plt # for plotting
import pandas as pd # for importing and managing datasets

#importing the dataset
dataset = pd.read_csv('Data.csv')

# independent variables = Country, Age, and Salary
# dependent variable = Purchased

X = dataset.iloc[:, :-1].values # all columns, except for the last one
Y = dataset.iloc[:, 3].values # dependent variable: purchased

"""
### HANDLE MISSING DATA [don't need] ###
#from sklearn.preprocessing import Imputer # CLASS: SK learn contains libraries to make machine models and preprocess datasets
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # Ctrl+i
#imputer = imputer.fit(X[:, 1:3]) # want index columns 1 and 2. Upper-bound is excluded (index 3)
#X[:, 1:3] = imputer.transform(X[:, 1:3]) # insert replacement for missing data in columns 1 and 2
#print(X)

### ENCODE VARIABLES INTO NUMBERS [don't need] - important for processing data in machine learning
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# print(X)

# expand three variables of single column into three separate columns
# DUMMY ENCODING
onehotencoder = OneHotEncoder(categorical_features = [0]) # encoded in alphabetical order left-to-right
X = onehotencoder.fit_transform(X).toarray()
#print(X)

# encode the dependent variables
labelencoder_Y = LabelEncoder()
dummyDependent = labelencoder_Y.fit_transform(Y)
"""

### Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummyDependent, test_size = .2, random_state = 0) # good size for test is usually .2. random_state is so that I have identical results as the instructor

# Build machine by using train variables to find correlations between independent and dependent data

"""
### FEATURE SCALING - ***SOME LIBRARIES APPLY THIS AUTOMATICALLY*** Age variable and Salary variables have different ranges. 
# Euclidean Distance between points. Scale the variables closer together; otherwise, one will dominate the other. 
# Putting variables in the same range. Standardization vs normalization.
# Even though most machine models are not based on Euclidean distances, scaling still helps the algorithm converge faster 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # create object of StandardScaler class
X_train = sc_X.fit_transform(X_train) # fit AND THEN transform
X_test = sc_X.transform(X_test) # don't need to fit because it already is..?
"""


