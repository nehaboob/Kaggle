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

with macro LB: giving -ve results
without macro LB: .31427

with macro LB: .31941

This exact verion gives LB: .31941

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import xgboost as xgb
import datetime

## read the files
train = pd.read_csv('../data/train.csv', parse_dates=['timestamp'])
macro = pd.read_csv('../data/macro.csv', parse_dates=['timestamp'])
test = pd.read_csv('../data/test.csv', parse_dates=['timestamp'])

id_test = test.id

# remove empty macro 
percent_null = macro.isnull().mean(axis=0) > 0.40
macro = macro.loc[:, ~percent_null]

## merge macro and house data
train = pd.merge(train, macro, how='left', on='timestamp')
test = pd.merge(test, macro, how='left', on='timestamp')
print(train.shape)
train.head()

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

# month and week count - not working
'''
month = (train.timestamp.dt.month)
month_cnt_map = month.value_counts().to_dict()
train['month_cnt'] = month.map(month_cnt_map)

month = (test.timestamp.dt.month)
month_cnt_map = month.value_counts().to_dict()
test['month_cnt'] = month.map(month_cnt_map)

# Add week-year count
week = (train.timestamp.dt.weekofyear)
week_cnt_map = week.value_counts().to_dict()
train['week_cnt'] = week.map(week_cnt_map)

week = (test.timestamp.dt.weekofyear)
week_cnt_map = week.value_counts().to_dict()
test['week_cnt'] = week.map(week_cnt_map)
'''

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
# Other feature engineering
train['rel_floor'] = 0.05+train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = 0.05+train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = 0.05+test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = 0.05+test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

train['area_per_room'] = train['life_sq'] / train['num_room'].astype(float) #rough area per room
train['livArea_ratio'] = train['life_sq'] / train['full_sq'].astype(float) #rough living area
train['yrs_old'] = 2017 - train['build_year'].astype(float) #years old from 2017
train['avgfloor_sq'] = train['life_sq']/train['max_floor'].astype(float) #living area per floor
train['pts_floor_ratio'] = train['public_transport_station_km']/train['max_floor'].astype(float)
# looking for significance of apartment buildings near public t 
train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
# doubled a var by accident
# when removing one score did not improve...
train['gender_ratio'] = train['male_f']/train['female_f'].astype(float)
train['kg_park_ratio'] = train['kindergarten_km']/train['park_km'].astype(float) #significance of children?
train['high_ed_extent'] = train['school_km'] / train['kindergarten_km'] #schooling
train['pts_x_state'] = train['public_transport_station_km'] * train['state'].astype(float) #public trans * state of listing
train['lifesq_x_state'] = train['life_sq'] * train['state'].astype(float) #life_sq times the state of the place
train['floor_x_state'] = train['floor'] * train['state'].astype(float) #relative floor * the state of the place

test['area_per_room'] = test['life_sq'] / test['num_room'].astype(float)
test['livArea_ratio'] = test['life_sq'] / test['full_sq'].astype(float)
test['yrs_old'] = 2017 - test['build_year'].astype(float)
test['avgfloor_sq'] = test['life_sq']/test['max_floor'].astype(float) #living area per floor
test['pts_floor_ratio'] = test['public_transport_station_km']/test['max_floor'].astype(float) #apartments near public t?
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)
test['gender_ratio'] = test['male_f']/test['female_f'].astype(float)
test['kg_park_ratio'] = test['kindergarten_km']/test['park_km'].astype(float)
test['high_ed_extent'] = test['school_km'] / test['kindergarten_km']
test['pts_x_state'] = test['public_transport_station_km'] * test['state'].astype(float) #public trans * state of listing
test['lifesq_x_state'] = test['life_sq'] * test['state'].astype(float)
test['floor_x_state'] = test['floor'] * test['state'].astype(float)


# check for the null value columns -- find out a proper imputation strategy
percent_null = train.isnull().mean(axis=0) > 0.40
print("{:.2%} of columns have more than 40% missing values.".format(np.mean(percent_null)))

# remove empty columns
train = train.loc[:, ~percent_null]
test = test.loc[:, ~percent_null]

print(train.shape)


# predict log of price
target = train['price_doc']
y = target

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

target.plot(ax=axes[0], kind='hist', bins=100)
np.log(target).plot(ax=axes[1], kind='hist', bins=100, color='green', secondary_y=True)
plt.show()


x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

# check type of objects 
num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))

f_train = x_all[:num_train]
f_test = x_all[num_train:]
# set our training data
X = f_train
x_test = f_test

# split the data into train and validation set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

june_index = train.timestamp.dt.date >= datetime.date(2014, 6, 30)
pre_june_index = train.timestamp.dt.date < datetime.date(2014, 6, 30)
X_train = X[pre_june_index]
y_train = y[pre_june_index]
X_test = X[june_index]
y_test = y[june_index]
# cost function to be minimized

def rmsle_exp(y_true_log, y_pred_log):
    y_true = y_true_log
    y_pred = y_pred_log
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

def score_model(model):
    train_error = rmsle_exp(y_train, model.predict(xgb.DMatrix(X_train)))
    test_error = rmsle_exp(y_test, model.predict(xgb.DMatrix(X_test)))
    return train_error, test_error

# fit the data to random forest regression
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 385  # This was the CV output, as earlier version shows
rfr = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
print("Train error: {:.4f}, Test error: {:.4f}".format(*score_model(rfr)))

# generate the test file
# Refit the model on everything, including our held-out test set.
dtrainf = xgb.DMatrix(X, y)
final = xgb.train(dict(xgb_params, silent=0), dtrainf, num_boost_round= num_boost_rounds)

y_predict = final.predict(dtest)
print("Train error: {:.4f}, Test error: {:.4f}".format(*score_model(final)))

gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

###########################################
print('Running Model 2...')
train = pd.read_csv('../data/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../data/test.csv', parse_dates=['timestamp'])

id_test = test.id

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 385  # This was the CV output, as earlier version shows
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


# Now, output it to CSV

first_result = output.merge(gunja_output, on="id", suffixes=['_louis','_bruno'])
first_result["price_doc"] = ( 0.5*first_result.price_doc_louis +
                                    0.5*first_result.price_doc_bruno ) 


first_result.drop(["price_doc_louis","price_doc_bruno"],axis=1,inplace=True)
first_result.head()
first_result.to_csv('pred_combined.csv', index=False)

