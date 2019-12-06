#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:42:37 2017

@author: neha
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
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

y_train = train["y"]
train = train.drop('y', axis=1)

# with dummy variable train and test set
d_train = pd.get_dummies(train).astype(np.float64)
print(d_train.shape)

## test has 6 more dummy variables
d_test = pd.get_dummies(test).astype(np.float64)
print(d_test.shape)

# check type of objects 
# with label encoder train and test set
num_train = len(train)
conncat =  pd.concat([train, test])

for c in conncat.columns:
    if conncat[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(conncat[c].values))
        conncat[c] = lbl.transform(list(conncat[c].values))

l_train = conncat[:num_train]
l_test = conncat[num_train:]


# check type of objects 
# with OnehoT encoder train and test set
num_train = len(train)
conncat =  pd.concat([train.drop('ID', axis =1), test.drop('ID', axis =1)])

for c in conncat.columns:
    if conncat[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(conncat[c].values))
        conncat[c] = lbl.transform(list(conncat[c].values))

o_train = conncat[:num_train]
o_test = conncat[num_train:]

onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3, 4, 5, 6, 7])
conn = onehotencoder.fit(conncat)
o_train = onehotencoder.transform(o_train).toarray()
o_test = onehotencoder.transform(o_test).toarray()

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
cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4,5], 'n_estimators': [100, 200]}


# version 1 - label encoder 
# {'colsample_bytree': 0.6, 'gamma': 0.3, 'max_depth': 2, 'min_child_weight': 5, 'subsample': 1.0}
# 0.5640753347054364
# LB 0.55742

xgb_model = xgb.XGBRegressor()
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM.fit(l_train, y_train)

#print(r2_score(Y_Val, grid.best_estimator_.predict(X_Val))) 
print(optimized_GBM.best_score_)
print(optimized_GBM.best_params_)

y_test = optimized_GBM.best_estimator_.predict(l_test)
results_df = pd.DataFrame(data={'y':y_test}) 
ids = test["ID"]
joined = pd.DataFrame(ids).join(results_df)
joined.to_csv("mercedes.csv", index=False)

# version 2 - label encoder - drop ID
# {'colsample_bytree': 0.6, 'gamma': 0.3, 'max_depth': 2, 'min_child_weight': 5, 'subsample': 1.0}
# 0.5592581546555323

l_train = l_train.drop('ID', axis =1)
l_test = l_test.drop('ID', axis = 1)

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM.fit(l_train, y_train)

print(optimized_GBM.best_score_)
print(optimized_GBM.best_params_)
plt.plot(train.ID, train.y, 'ro')

# version 3 - pandas dummies - drop ID
# {'colsample_bytree': 1.0, 'gamma': 0.3, 'max_depth': 2, 'min_child_weight': 5, 'subsample': 1.0}
# 0.563978686336157
# LB 0.55044

d_train = d_train.drop('ID', axis =1)
d_test = d_test.drop('ID', axis = 1)

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM.fit(d_train, y_train)

print(optimized_GBM.best_score_)
print(optimized_GBM.best_params_)


# Make sure it's in the same format as the training data
df_test = pd.DataFrame(columns=d_train.columns)
for column in df_test.columns:
    if column in d_test.columns:
        df_test[column] = d_test[column]
    else:
        df_test[column] = np.nan

y_test = optimized_GBM.best_estimator_.predict(df_test)
results_df = pd.DataFrame(data={'y':y_test}) 
ids = test["ID"]
joined = pd.DataFrame(ids).join(results_df)
joined.to_csv("mercedes.csv", index=False)

plt.plot(joined.ID, joined.y, 'ro')

# version 4 - onehot encoder - drop ID
#0.5640839654519466
#{'colsample_bytree': 0.9, 'gamma': 0.3, 'max_depth': 2, 'min_child_weight': 5, 'subsample': 1.0}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM.fit(o_train, y_train)

print(optimized_GBM.best_score_)
print(optimized_GBM.best_params_)

y_test = optimized_GBM.best_estimator_.predict(o_test)
results_df = pd.DataFrame(data={'y':y_test}) 
ids = test["ID"]
joined = pd.DataFrame(ids).join(results_df)
joined.to_csv("mercedes.csv", index=False)
0.5894585815056219

# version 5 - version 4 with more grid search
#0.5640839654519466
#{'colsample_bytree': 0.9, 'gamma': 0.1, 'max_depth': 2, 'min_child_weight': 5, 'subsample': 1.0}

cv_params = {'min_child_weight':[4,5,6], 'gamma':[i/10.0 for i in range(1,4)],  'subsample':[i/10.0 for i in range(9,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [1,2,3], 'n_estimators': [200]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM.fit(o_train, y_train)

print(optimized_GBM.best_score_)
print(optimized_GBM.best_params_)
print(optimized_GBM.grid_scores_)


#plot the feature impotance
xgb.plot_importance(optimized_GBM.best_estimator_)
reg = optimized_GBM.best_estimator_.booster().get_fscore()

sorted(reg.items(), key=lambda x:x[1])

imp_columns = []
for key in reg:
    if(reg[key] >= 5):
        imp_columns.append(key)
    
# feature selection using xgb
model = xgb.XGBRegressor(colsample_bytree= 1.0, gamma=0.3, max_depth=2, min_child_weight= 5, subsample= 1.0)
model.fit(d_train, y_train)
thresholds = model.booster().get_fscore().values()


## fit on sected features  
#0.572283586001696
#LB  0.54432
s_train = d_train[imp_columns]
s_test =  d_test[imp_columns] 

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM.fit(s_train, y_train)

print(optimized_GBM.best_score_)
print(optimized_GBM.best_params_)


y_test = optimized_GBM.best_estimator_.predict(s_test)
results_df = pd.DataFrame(data={'y':y_test}) 
ids = test["ID"]
joined = pd.DataFrame(ids).join(results_df)
joined.to_csv("mercedes.csv", index=False)
