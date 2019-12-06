#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:25:15 2017

@author: neha

#split the data on X322

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

#remove outlier
out = Y[Y > 125].index.values  
X_out = X.drop(out, axis=0)
Y_out = Y.drop(out, axis=0)


x_tr, x_val, y_tr, y_val = train_test_split(X, Y, test_size=0.2, random_state = 108)

out_tr = y_tr[y_tr > 125].index.values  
out_val =  y_val[y_val > 125].index.values  
x_out_tr = x_tr.drop(out_tr, axis = 0)
y_out_tr = y_tr.drop(out_tr, axis = 0)

var = X.X232
plt.hist(Y[var==0], 200, alpha=0.5, label='0', color = plt.cm.jet(0))
plt.hist(Y[var==1], 200, alpha=0.75, label='1', color = plt.cm.jet(255))
plt.title('X232')

index_232_0 = Y[var==0].index.values #3739
index_232_1 = Y[var==1].index.values #172
              
# apply XGB on small values x232 == 1

X_232 = X.drop(index_232_0, axis=0)
Y_232 = Y.drop(index_232_0, axis=0)

# exp1 -- with outlier XGB
def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print('MSE: %2.3f' % mse)
    return mse

def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     print('R2: %2.3f' % r2)
     return r2

def two_score(y_true,y_pred):    
    #MSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    score = R2(y_true,y_pred)
    return score

def two_scorer():
    return make_scorer(two_score, greater_is_better=True) # change for false if using MSE



cv_params = {'min_child_weight':[1, 2, 3, 4], 'gamma':[i/10.0 for i in range(1,5)],  'subsample':[i/10.0 for i in range(1,3)],
'colsample_bytree':[i/10.0 for i in range(1,11)], 'max_depth': [1,2,3], 'n_estimators': [500]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(X_232, Y_232)

print('r2 score', r2_score(Y_232, optimized_GBM.best_estimator_.predict(X_232))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)
               
##optimized_GBM.grid_scores_
plt.hist(Y[var==1], 200, alpha=0.75, label='1', color = plt.cm.jet(255))
plt.hist(optimized_GBM.best_estimator_.predict(X_232), 200, alpha=0.5, label='0', color = plt.cm.jet(0))


from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
X_232_pca = pca.fit_transform(X_232)
explained_variance = pca.explained_variance_ratio_.cumsum()

plt.plot(explained_variance)


cv_params = {'min_child_weight':[1, 2, 3, 4], 'gamma':[i/10.0 for i in range(1,5)],  'subsample':[i/10.0 for i in range(1,3)],
'colsample_bytree':[i/10.0 for i in range(1,11)], 'max_depth': [1,2,3], 'n_estimators': [500]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(X_232_pca, Y_232)

print('r2 score', r2_score(Y_232, optimized_GBM.best_estimator_.predict(X_232_pca))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

##optimized_GBM.grid_scores_
plt.hist(Y[var==1], 200, alpha=0.75, label='1', color = plt.cm.jet(255))
plt.hist(optimized_GBM.best_estimator_.predict(X_232_pca), 200, alpha=0.5, label='0', color = plt.cm.jet(0))

