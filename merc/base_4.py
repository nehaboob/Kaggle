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

'''
# apply PCA and then XGboost
#applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 222)
X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_.cumsum()

plt.plot(explained_variance)


# apply the K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 20, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X_pca)
y_test_kmeans = kmeans.fit_predict(X_test_pca)
plt.plot(y_kmeans, Y, 'ro')

X_k= X
X_test_k = X_test
X_k['KnnClu'] = y_kmeans
X_test_k['KnnClu'] = y_test_kmeans
      
# apply XGBoost
#r2 score 0.460871832664
#CV score 0.6041901782313058
#r2 score 0.463405623889
#CV score 0.6102186858405328

x_tr, x_val, y_tr, y_val = train_test_split(X_k, Y, test_size=0.2)

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


y_test = optimized_GBM.best_estimator_.predict(X_test_k)

plt.hist(y_test, 200)

sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = y_test
sub.to_csv('xgb_full.csv', index=False)

'''

## Autoencoder Clusters
# applying autoencoder
from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 16
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

# let's create and compile the autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.2, random_state=42)

# these parameters seems to work for the Mercedes dataset
history = autoencoder.fit(X.values, X.values,
                epochs=500,
                batch_size=200,
                shuffle=False,
                verbose = 2,
                validation_data=None)


plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# now let's evaluate the coding of the initial features
encoder = Model(input_layer, encoded)
preds_x = encoder.predict(X.values)
preds_x_test = encoder.predict(X_test.values)

# apply the K-means on PCA
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 20, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(preds_x)
y_test_kmeans = kmeans.fit_predict(preds_x_test)


plt.plot(y_kmeans, Y, 'ro')

## add clustyer variable to thte data set
X_k= X.copy()
X_test_k = X_test.copy()
X_k['KnnClu'] = y_kmeans
X_test_k['KnnClu'] = y_test_kmeans
        
        
# apply XGBoost
#r2 score 0.622721394709
#CV score 0.5613276298297174
#r2 score 0.657167492537
#CV score 0.5569004418861734
#r2 score 0.514094492643
#CV score 0.5802906241619604
#r2 score 0.479898028707
#CV score 0.5952151234600774
x_tr, x_val, y_tr, y_val = train_test_split(X_k, Y, test_size=0.2)

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

optimized_GBM.fit(X_k, Y)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

y_test = optimized_GBM.best_estimator_.predict(X_test_k)

plt.hist(y_test, 200)

sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = y_test
sub.to_csv('xgb_full.csv', index=False)

#r2 score 0.67398179245
#CV score 0.6276880327941153

out = Y[Y > 125].index.values  # Approximately 0.02% of the data or 150 / 125?
X_dr = X_k.drop(out, axis=0)
Y_dr = Y.drop(out, axis=0)


xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(X_dr, Y_dr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

y_test_dr = optimized_GBM.best_estimator_.predict(X_test_k)


y_avg = (y_test+y_test_dr)/2
        
plt.hist(y_test, 200)
plt.hist(y_test_dr, 200)
       
        
        
sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = y_avg
sub.to_csv('xgb_full_aqvg.csv', index=False)
        

'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

plt.scatter(y_test_dr, y_test, color='red')
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')


'''
#predict the new observation
y_pred = regressor.predict(X_test)
