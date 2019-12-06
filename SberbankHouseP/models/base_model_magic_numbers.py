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
train = pd.read_csv('../data/train.csv', parse_dates=['timestamp'])
macro = pd.read_csv('../data/macro.csv', parse_dates=['timestamp'])
test = pd.read_csv('../data/test.csv', parse_dates=['timestamp'])

## merge macro and house data
train = pd.merge(train, macro, how='left', on='timestamp')
test = pd.merge(test, macro, how='left', on='timestamp')
print(train.shape)
train.head()

# plot price of House
target = train['price_doc']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

target.plot(ax=axes[0], kind='hist', bins=100)
np.log(target).plot(ax=axes[1], kind='hist', bins=100, color='green', secondary_y=True)
plt.show()

#clean data
bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
test.loc[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index
test.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.loc[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.loc[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.loc[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = np.NaN
test.state.value_counts()

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc/train.full_sq <= 600000]
train = train[train.price_doc/train.full_sq >= 10000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

# check for the null value columns -- find out a proper imputation strategy
percent_null = train.isnull().mean(axis=0) > 0.20
print("{:.2%} of columns have more than 20% missing values.".format(np.mean(percent_null)))

# remove uninformative columns
#df = train.loc[:, ~percent_null]
#df = train.drop(['id', 'price_doc'], axis=1)
#print(df.shape)


x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
# predict log of price
target = train['price_doc']
y = np.log(target)
y = target
#x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

# check type of objects 
print(x_train.dtypes.value_counts())
np.array([c for c in x_train.columns if x_train[c].dtype == 'object'])

# dummy variables for objects
#x_train['timestamp'] = pd.to_numeric(pd.to_datetime(x_train['timestamp']))  / 1e18
#print(x_train['timestamp'].head())
# This automatically only dummies object columns
x_train = pd.get_dummies(x_train).astype(np.float64)
print(x_train.shape)

# set our training data
X = x_train

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
    #y_true = np.exp(y_true_log)
    #y_pred = np.exp(y_pred_log)
    y_true = y_true_log
    y_pred = y_pred_log
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
test_data = x_test
#test_data['timestamp'] = pd.to_numeric(pd.to_datetime(test_data['timestamp'])) / 1e18
test_data = pd.get_dummies(test_data).astype(np.float64)

# Make sure it's in the same format as the training data
df_test = pd.DataFrame(columns=x_train.columns)
for column in df_test.columns:
    if column in test_data.columns:
        df_test[column] = test_data[column]
    else:
        df_test[column] = np.nan

# Make the predictions
pred = rfr.predict(pipe.transform(df_test))
#predictions = np.exp(rfr.predict(pipe.transform(df_test)))
predictions = rfr.predict(pipe.transform(df_test))
# And put this in a dataframe
predictions_df = pd.DataFrame()
predictions_df['id'] = test['id']
predictions_df['price_doc'] = predictions
predictions_df.head()

# Now, output it to CSV
predictions_df.to_csv('predictions.csv', index=False)
