#%%
#necessary imports
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error



#%% 
"""GETTING THE DATA"""
def getData(path):
    train = pd.read_csv(f"{path}/train.csv")
    test = pd.read_csv(f"{path}/test.csv")
    y = train["emission"]
    train.drop("emission", axis = 1, inplace = True)

    return train, test, y
# %%
"""DATA SCALING"""

def Znormalize(train, test):
    """
    CONTRACT:
    Znormalize(train, test) -> new_train, new_test
    train: pandas dataframe that contains the training data
    test: pandas dataframe that contains the testing data

    new_train, new_test: same as above but z-normalized

    DESCRIPTION:
    This function takes in both the training and testing data and then 
    combines it into a trainplustest dataframe. It then uses that dataframe to 
    znormalize both the train and test dataframes. It then returns the new dataframes
    """
    trainplustest = pd.concat([train, test], ignore_index=True)
    new_train = train.copy()
    new_test = test.copy()
    for column in trainplustest.columns:
        print(column)
        if type(trainplustest[column][1]) != str:
            col_mean = trainplustest[column].mean(numeric_only = False)
            col_std = trainplustest[column].std()
            if col_std != 0:
                new_train[column] = (train[column]  - col_mean)/col_std
                new_test[column] = (test[column]  - col_mean)/col_std
        else:
            new_test[column] = test[column]
            new_train[column] = train[column]


    return new_train, new_test

def reverse_Znormalize(train, test, normalized_data):
    """
    Reverse the normalization process
    """
    normalized_data = normalized_data.copy()
    unnormalized_data = pd.concat([train, test], ignore_index=True)
    for column in unnormalized_data.columns:
        if type(unnormalized_data[column][0]) != str:
            col_mean = unnormalized_data[column].mean()
            col_std = unnormalized_data[column].std()
            normalized_data[column] = normalized_data[column] * col_std + col_mean
        else:
            normalized_data[column] = unnormalized_data[column]

    return normalized_data

# %%
"""DATA IMPUTATION"""


def predictMissingValues(train_data, test_data):
    """
    CONTRACT
    predictMissingValues(train_data, test_data) -> filled_train, filled_test
    train_data: pandas dataframe that contains the training data
    test_data: pandas dataframe that contains the testing data

    filled_train, filled_test: same as above but with the missing values filled in through 
    a KNN model

    DESCRIPTION
    
    Given the data, predict the missing values given a k nearest neightors regressor

    This will fill in the values from the latitude, logitude, and year 
    (as those don't have any missing values)

    Also, this will print out the error for each columun 
    """
    #for each column with missing rows of data, split the the columns into rows with the missing values 
    # and those without the missing values

    #get the columns with the missing values
    trainplustest = pd.concat([train_data, test_data])

    columns_with_missing_values = trainplustest.columns[trainplustest.isnull().any()]
    filled_train = train_data.copy()
    filled_test = test_data.copy()

    for column in columns_with_missing_values:
        #split the rows into those with missing values and those without
        train_with_missing_values = train_data[train_data[column].isnull()]
        test_with_missing_values = test_data[test_data[column].isnull()]
        rows_without_missing_value = trainplustest[trainplustest[column].notnull()]

        #get the longitude, latitude, year, and column with the missing values
        X = rows_without_missing_value[['longitude', 'latitude', 'year']]
        y = rows_without_missing_value[[column]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = KNeighborsRegressor(n_neighbors=5)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"Error for column {column}: {mean_squared_error(y_test, y_pred)}")

        #predict the missing values
        model.fit(X, y)
        predicted_values_test = model.predict(test_with_missing_values[['longitude', 'latitude', 'year']])
        #fill in the missing values
        for i in range(predicted_values_test.shape[0]):
            pd.options.mode.chained_assignment = None
            filled_test[column][test_with_missing_values.index[i]] = predicted_values_test[i][0]

        predicted_values_train = model.predict(train_with_missing_values[['longitude', 'latitude', 'year']])
        for i in range(predicted_values_train.shape[0]):
            pd.options.mode.chained_assignment = None
            filled_train[column][train_with_missing_values.index[i]] = predicted_values_train[i][0]
        

    return filled_train, filled_test


#%%

