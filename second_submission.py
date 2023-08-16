"""
SECOND SUBMISSION TO CONTEST

After seeing the best score from the last model I've decided to imporve everything by adding
and XGBoost model to the mix. I think that through structured data, a combination of XGBoost and 
NN would help it suceed.

Besides that, it's really important to change how the data is selected. I'm probably going to look into ways of 
imputation beyond KNN also select based on highest correlation, not just lowest missing values.
"""

# %%
"""IMPORTS

Same imports as from the first submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


#import utils
import utils

# %%
"""Getting the DATG"""

PATH = "playground-series-s3e20"
train, test, y = utils.getData(PATH)
print("Traing Dataset")
display(train.head())

print("Testing Dataset")
display(test.head())

print("y")
display(y.head())


# %%
"""
DATA Imputation
I'm going to imput data first, so it's easier to choose 
between the different columns
"""