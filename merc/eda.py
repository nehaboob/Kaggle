#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:32:09 2017

@author: neha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

y = train['y']
train = train.drop('y', 1)

conncat =  pd.concat([train, test])

out = y[y > 250].index.values  # Approximately 0.02% of the data or 150 / 125?
X = train.drop(out, axis=0)
Y = y.drop(out, axis=0)




plt.plot(train.ID, train.y, 'ro')
plt.hist(X.y, 300)
plt.bar(train.ID, train.y)
plt.plot(train.y)

i = 0
# constant columns in training set
for col in train:
    #if(len(train[col].unique()) == 1):
    if(i<10):
        print(col)
        print(len(train[col].unique()))
        print(len(test[col].unique()))
        i=i+1
        
# constant columns in test set
for col in test:
    if(len(test[col].unique()) == 1):
        print(col)
        print(test[col].unique())        
        
# constant columns in concatenated set
for col in conncat:
    if(len(conncat[col].unique()) == 1):
        print(col)
        print(conncat[col])
        

# find duplicates
train = train.drop('ID', axis = 1)

duplicate_rows = train[train.duplicated(keep = False)]

y[duplicate_rows.index]


## from kernal 
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

feature_columns = train_data.columns[2:]

label_columns = []
for dtype, column in zip(train_data.dtypes, train_data.columns):
    if dtype == object:
        label_columns.append(column)


print("{} duplicate entries in training, out of {}, a {:.2f} %".format(
    len(train_data[train_data.duplicated(subset=feature_columns, keep=False)]),
    len(train_data),
    100 * len(train_data[train_data.duplicated(subset=feature_columns, keep=False)]) / len(train_data)
    ))

# list of all duplicate rows
train_data[train_data.duplicated(subset=feature_columns, keep=False)].sort_values(by=label_columns)

# standard deviation of the group
duplicate_std = train_data[train_data.duplicated(subset=feature_columns,
                             keep=False)].groupby(list(feature_columns.values))['y'].aggregate(['std', 'size']).reset_index(drop=True)

duplicate_std.sort_values(by='std', ascending=False)

print("{} duplicate groups in training".format(
    len(train_data[train_data.duplicated(subset=feature_columns,
                             keep=False)].groupby(list(feature_columns.values)).size().reset_index())))

    
train_data[train_data.duplicated(subset=feature_columns,
                             keep=False)].groupby(list(feature_columns.values)).size().reset_index()



'''
Notes:
    1. THere are some features values which are constant in train and some are constant in test
    2. categorical variables to numeric variables
    3. We need to find out way to ahndel duplicate rows
    4. we need to find of corelating columns
'''