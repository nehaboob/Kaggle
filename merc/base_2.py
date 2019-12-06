#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:42:37 2017

@author: neha

data cleaning:- remove outliers and duplicate rows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import grid_search
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Ridge
import xgboost as xgb
from numpy import sort
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# one hot encoding
data = train_df.append(test_df)
data = pd.get_dummies(data)
train, test = data[0:len(train_df)], data[len(train_df):]

# data for training
X = train.drop(["y", "ID"], axis=1)
Y = train["y"]
X_test = test.drop(["y", "ID"], axis=1)

# handel duplicates, replace Y with mean of the duplicate and then remove these rows
def average_dupes(x):
    print(x.index)
    print(Y.loc[list(x.index)].mean())
    Y.loc[list(x.index)] = Y.loc[list(x.index)].mean()

dupes = X[X.duplicated(keep = False)]
dupes.groupby(list(dupes.columns)).apply(average_dupes)

# remove deplucate rows
dupe_indexes = X[X.duplicated()].index.values
X = X.drop(dupe_indexes, axis=0)
Y = Y.drop(dupe_indexes, axis=0)


out = Y[Y > 150].index.values  # Approximately 0.02% of the data or 150 / 125?
X = X.drop(out, axis=0)
Y = Y.drop(out, axis=0)

x_tr, x_val, y_tr, y_val = train_test_split(X, Y, test_size=0.2)


# two scorer

def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print('MSE: %2.3f' % mse)
    return mse

def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     print('R2: %2.3f' % r2)
     return r2

def two_score(y_true,y_pred):    
    MSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    score = R2(y_true,y_pred)
    return score

def two_scorer():
    return make_scorer(two_score, greater_is_better=True) # change for false if using MSE

# params for grid search with CV
cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}


# version 4 - onehot encoder - drop ID
#{'colsample_bytree': 1.0, 'gamma': 0.3, 'max_depth': 2, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 1.0}
#CV score 0.5935945800977694  -- ouliier 250
#LB 0.55302 -- ouliier 250
#CV score  0.709982158177095 -- ouliier 125
#r2 score -- 0.734530902608 ouliier 125
#r2 score 0.620979775314 ouliier 150
#CV score 0.6360437034797325 ouliier 150

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(X, Y)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)


y_test = optimized_GBM.best_estimator_.predict(X_test)

plt.hist(y_test, 200)

sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = y_test
sub.to_csv('xgb_full.csv', index=False)


