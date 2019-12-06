#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:48:22 2017

@author: nehaboob

I did first submission for this competition, following below kernel, got LB score 0.33180

https://www.kaggle.com/mwaddoups/i-regression-workflow-various-models 
High level approach is:

1. Merge Marco and Housing data on timestamp.
2. Remove features which are more than 20% null
3. Convert categorical features into dummy features (One hot encoding :P)
4. Fit the data to model. I used Random forests.
5. Predict log of housing price
6. Prepare the test data and predict results.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## read the files
train = pd.read_csv('../data/train.csv')
macro = pd.read_csv('../data/macro.csv')
test = pd.read_csv('../data/test.csv')

## merge macro and house data
train = pd.merge(train, macro, how='left', on='timestamp')
print(train.shape)
train.head()

# plot price of House
target = train['price_doc']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

target.plot(ax=axes[0], kind='hist', bins=100)
np.log(target).plot(ax=axes[1], kind='hist', bins=100, color='green', secondary_y=True)
plt.show()

# predict log of price
y = np.log(target)

# check for the null value columns -- find out a proper imputation strategy
percent_null = train.isnull().mean(axis=0) > 0.20
print("{:.2%} of columns have more than 20% missing values.".format(np.mean(percent_null)))

# remove uninformative columns
df = train.loc[:, ~percent_null]
df = df.drop(['id', 'price_doc'], axis=1)
print(df.shape)

# check type of objects 
print(df.dtypes.value_counts())
np.array([c for c in df.columns if df[c].dtype == 'object'])

# dummy variables for objects
df['timestamp'] = pd.to_numeric(pd.to_datetime(df['timestamp']))  / 1e18
print(df['timestamp'].head())
# This automatically only dummies object columns
df = pd.get_dummies(df).astype(np.float64)
print(df.shape)

# set our training data
X = df

# split the data into train and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# preprocess data
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import make_pipeline

# Make a pipeline that transforms X
pipe = make_pipeline(Imputer(), StandardScaler())
pipe.fit(X_train)
pipe.transform(X_train)

# cost function to be minimized

def rmsle_exp(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

def score_model(model, pipe):
    train_error = rmsle_exp(y_train, model.predict(pipe.transform(X_train)))
    test_error = rmsle_exp(y_test, model.predict(pipe.transform(X_test)))
    return train_error, test_error

# fit the data to random forest regression
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, n_jobs=-1)
rfr.fit(pipe.transform(X_train), y_train)

print("Train error: {:.4f}, Test error: {:.4f}".format(*score_model(rfr, pipe)))

# generate the test file
# Refit the model on everything, including our held-out test set.
pipe.fit(X)
rfr.fit(pipe.transform(X), y)
print("Train error: {:.4f}, Test error: {:.4f}".format(*score_model(rfr, pipe)))

# Apply the same steps to process the test data
test_data = pd.merge(test, macro, how='left', on='timestamp')
test_data['timestamp'] = pd.to_numeric(pd.to_datetime(test_data['timestamp'])) / 1e18
test_data = pd.get_dummies(test_data).astype(np.float64)

# Make sure it's in the same format as the training data
df_test = pd.DataFrame(columns=df.columns)
for column in df_test.columns:
    if column in test_data.columns:
        df_test[column] = test_data[column]
    else:
        df_test[column] = np.nan

# Make the predictions
pred = rfr.predict(pipe.transform(df_test))
predictions = np.exp(rfr.predict(pipe.transform(df_test)))

# And put this in a dataframe
predictions_df = pd.DataFrame()
predictions_df['id'] = test['id']
predictions_df['price_doc'] = predictions
predictions_df.head()

# Now, output it to CSV
predictions_df.to_csv('predictions.csv', index=False)
