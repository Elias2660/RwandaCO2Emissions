
"""
FIRST SUBMISSION


This file will use a simple neural network that is scaled to predict the C02 Emissions of 
places and times in Rwanda

"""

#%%
#NECESSARY IMPORTS

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


print("Completed Imports")
#%%


PATH = "playground-series-s3e20"

train = pd.read_csv(f"{PATH}/train.csv")
test = pd.read_csv(f"{PATH}/test.csv")
y = train["emission"]
train.drop("emission", axis = 1, inplace = True)

#%%
#displaying the training and testing data
display(train.head())
print(f"Shape of the Training Data: {train.shape}")
print(f"shape of the Testing Data: {test.shape}")



# %%
"""DATA SCALING"""
"""


I need to find out ways of scaling the data so one can get a good prediction

I'm thinking of z-score normalization
"""

def normalize(train, test):
    """
    Reverse the normalization process
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

def reverse_normalize(train, test, normalized_data):
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


znormalized_train, znormalized_test = normalize(train, test)
znormalized_y = (y - y.mean())/y.std()
print(f"Normalized Training data:")
display(znormalized_train.head())
print(f"Train Data: ")
display(train.head())
print(f"Unormalized Training Data:")
display(reverse_normalize(train, test, znormalized_train).head())
print(f"Normalized Testing Data:")
display(znormalized_test.head())


#%%
"""
PDA
"""
"""


I'm really intrigued about the missing values problem in the data.

I'm also interested in the distribution of the data based on the different features
"""

#checking for missing values
print(f"Rows with missing values in the training data: {train.isnull().sum()}")

#top 7 columns with missing values
missing_values_sorted = train.isnull().sum().sort_values(ascending = False).keys()

print(f"21 Columns with the lowest missing values{missing_values_sorted[missing_values_sorted.shape[0]-21:]}")

print(f"""Highest Number of Missing Values in the 21 cols with lowest missing values:
       {train[missing_values_sorted[missing_values_sorted.shape[0]-40:]].isnull().sum().max()}""")

#%%
"""We're going to use the 50 columns with lowest missing values to predict

Here, create new dataframes based on the columns that we're going to use
"""

missing_values_sorted = test.isnull().sum().sort_values(ascending = True).keys()
def topXcolumns_with_missing_values(X, missing_values_sorted):
    """
    Return the top X columns with the lowest missing values
    """
    return missing_values_sorted[:X]

new_train = znormalized_train[topXcolumns_with_missing_values(50, missing_values_sorted)]
new_test = znormalized_test[topXcolumns_with_missing_values(50, missing_values_sorted)]

print(f"New Train Data:")
display(new_train.head())
print(f"New Test Data:")
display(new_test.head())
# %%
"""SOLVING FOR MISSING VALUES"""
"""


I don't know how to solve the missing values problem

It could be useful to user a knn model to predict the missing values
"""

def predictMissingValues(train_data, test_data):
    """
    
    Given the data, predict the missing values given a k nearest neightors regressor

    This will fill in the values from the latitude, logitude, and year 
    (as those don't have any missing values)

    Also, this will print out the error for each co,mun 
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


final_train, final_test = predictMissingValues(new_train, new_test)
print("Final Train Data:")
display(final_train.head())
print("Final Test Data:")
display(final_test.head())

# %%
"""LAST THINGS  TO DO BEFORE MODELING"""

final_train_IDs = final_train["ID_LAT_LON_YEAR_WEEK"]
final_train.drop("ID_LAT_LON_YEAR_WEEK", axis = 1, inplace = True)
X_train, X_valid, y_train, y_valid = train_test_split(final_train, znormalized_y, test_size=0.2)

#turn X_train, X_valid, y_train, y_valid to float64

# %%
"""
MODEL INPORTS

This model will use a tensorflow model to 
predict the C02 emissions of places and times in Rwanda

Don't forget droupout layers, regularization, early stopping
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
INPUT_SHAPE = X_train.shape[1]

# %%

"""MODEL

"""
#create the model
R = 0.00001


model = Sequential([ 
    Dense(units = 512, input_shape = INPUT_SHAPE, activation = "relu"),
    Dense(units = 1024, activation = "relu", kernel_regularizer = l2(R)),
    Dense(units = 512, activation = "relu", kernel_regularizer = l2(R)),
    Dense(units = 512, activation = "relu", kernel_regularizer = l2(R)),
    Dense(units = 256, activation = "relu", kernel_regularizer = l2(R)),
    Dense(units = 128, activation = "relu", kernel_regularizer = l2(R)),
    Dense(units = 512, activation = "relu", kernel_regularizer = l2(R)),
    Dense(units = 64, activation = "relu", kernel_regularizer = l2(R)),
    Dense(units = 1)
])

model.compile(optimizer = "adam",
               loss = "mse",
                 metrics = ["mse", RootMeanSquaredError(name="root_mean_squared_error")])


early_stopping = EarlyStopping( 
    patience = 100, min_delta = 0.001, restore_best_weights = True
)

history = model.fit (
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    epochs = 1000,
    batch_size= 32,
    callbacks = [early_stopping]

)


#%%

"""
Plotting Curve of Model
"""
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="loss")
history_df.loc[:, ["root_mean_squared_error"]].plot(title="Accuracy")


#%%
"""
SUBMISSION
"""

#use model to predict the test data
final_test_IDs = final_test["ID_LAT_LON_YEAR_WEEK"]
y_preds = (model.predict(final_test.drop("ID_LAT_LON_YEAR_WEEK", axis = 1)) + y.mean())*y.std()


#%%
print(f"Predictions before reshaping: {y_preds.shape}")
y_preds = np.reshape(y_preds, newshape=(final_test_IDs.shape))
print(f"Predictions after reshaping: {y_preds.shape}")

# %%

final =  pd.DataFrame([final_test_IDs, y_preds]).transpose().rename(columns={"Unnamed 0": "emission"})
display(final)

# %%
submission = final.to_csv("first_submission.csv")


# %%
