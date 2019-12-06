#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 10:28:51 2017

@author: neha


adding clusters to the train data
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


# apply PCA and then XGboost
#applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_.cumsum()

plt.plot(explained_variance)


# apply the K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 20, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X_pca)
y_test_kmeans = kmeans.predict(X_test_pca)
plt.plot(y_kmeans, Y, 'ro')

X_k= X_pca.copy()
X_test_k = X_test_pca.copy()
X_k = np.column_stack((X_k, y_kmeans))
X_test_k = np.column_stack((X_test_k, y_test_kmeans))
      
# apply XGBoost
#r2 score 0.460871832664
#CV score 0.6041901782313058
#r2 score 0.463405623889
#CV score 0.6102186858405328

x_tr, x_val, y_tr, y_val = train_test_split(X_k, Y, test_size=0.2, random_state = 108)

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


xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(x_tr, y_tr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

reg = optimized_GBM.best_estimator_.booster().get_fscore()

val = pd.DataFrame()
val['y_val'] = y_val
val['y_pred'] = optimized_GBM.best_estimator_.predict(x_val)
val.to_csv('xgb_val.csv', index=False)

plt.hist(optimized_GBM.best_estimator_.predict(x_val), 200)

# create submission file
y_test = optimized_GBM.best_estimator_.predict(X_test_k)

plt.hist(y_test, 200)
plt.hist(train.y, 200)

sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = y_test
sub.to_csv('xgb_full.csv', index=False)



