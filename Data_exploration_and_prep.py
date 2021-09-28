
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix


# I use decimal=',' to make the commas into dots
train = pd.read_csv(r'C:\Users\alex_\Desktop\Kaggle\Disease prediction\Training.csv', decimal=',')
test = pd.read_csv(r'C:\Users\alex_\Desktop\Kaggle\Disease prediction\Testing.csv', decimal=',')


# Defining datasets
X_train = train.drop('prognosis', axis=1)
X_test = test.drop('prognosis', axis=1)

y_train = train['prognosis']
y_test = test['prognosis']


# Initial glance at data
print(X_train.shape)
print(X_train.columns)
print(X_train.info())
print(X_train.describe())
print(X_train.isnull().sum())
print(X_train.nunique())
print(X_train.dtypes)

# This tells us there are no missing values and all columns have binary values. 
# In other words, these columns are dummy variables indicating whether or not
# a patient has a certain symptom

# The last variable 'Unnamed: 133' seems odd and so I investigate further
# Because it has zero values I choose to drop it from the data
# Also its not mentioned by the author of the dataset so I assume it's an error

print(X_train['Unnamed: 133'].head())
print(X_train['Unnamed: 133'].value_counts())
print(X_train['Unnamed: 133'].describe())
X_train = X_train.drop(['Unnamed: 133'], axis=1)

# A quick check to see if the column "Unnamed: 133" exists in the test data as well. It doesnt
print(X_test.loc[:,'Unnamed: 133'])
print(X_test.columns)


# Initial look at the target variable. 
# We see here how the target variable consists of 41 different outcomes (illnesses) 
# and there is no missing
print(y_train.tail())
print(y_train.value_counts())
print(y_train.describe())
print(y_train.isnull().sum())
print(y_train.nunique())

# Second look at the target variable but for the test data
print(y_test.tail())
print(y_test.value_counts())
print(y_test.describe())
print(y_test.isnull().sum())
print(y_test.nunique())


# At this point the data is ready for modelling as it is clean, tidy, and without any missing or erroneous data
# My intention with it is to run a normal XGBoost classification with a binary target variable, and to run a multiclass XGBoost classification on it
# For the former goal I will first need to convert the target variable into a binary variable. For the second goal, I can simply leave the target variable as it is with its
# distinct values

# In order to convert the target variable into a binary dummy variable, I (a) create a copy of the target variable, (b) run a simple for-loop to flag the label og interest, and
# (c) repeat the process in order to have changed copies of both train and test data


z_train = y_train
z_train = z_train.to_frame()

result = []
for value in z_train["prognosis"]:
    if value == 'Pneumonia':           # Choice of Pneumonia is random
        result.append(1)
    else:
        result.append(0)
       
z_train["Result"] = result   
print(z_train["Result"].value_counts()) # Doubblechecking the figures to make sure the for-loop worked correctly

# Then repeat the process for the test data

z_test = y_test
z_test = z_test.to_frame()


result = []
for value in z_test["prognosis"]:
    if value == 'Pneumonia':
        result.append(1)
    else:
        result.append(0)
       
z_test["Result"] = result   
print(z_test["Result"].value_counts())

# Lastly I drop the original target variable "prognosis" from the y-axis data.

z_train = z_train.drop(['prognosis'], axis=1)
z_test = z_test.drop(['prognosis'], axis=1)






