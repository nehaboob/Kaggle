#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:42:37 2017

@author: neha

Apply clustering to the data
Apply PCA 

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
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_.cumsum()

plt.plot(explained_variance)


x_tr, x_val, y_tr, y_val = train_test_split(X_pca, Y, test_size=0.2)

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
# {'colsample_bytree': 0.8, 'gamma': 0.3, 'max_depth': 3, 'min_child_weight': 4, 'n_estimators': 100, 'subsample': 1.0}
#CV score 0.4933105616896883 
#LB 0.49478 

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(X_pca, Y)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

reg = optimized_GBM.best_estimator_.booster().get_fscore()


y_test = optimized_GBM.best_estimator_.predict(X_test)

plt.hist(y_test, 200)

sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = y_test
sub.to_csv('xgb_full.csv', index=False)


# cluster X values in 4 clusters

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 100):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,100), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

# apply the K-means
kmeans = KMeans(n_clusters = 14, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

plt.plot(y_kmeans, Y, 'ro')


#applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 222)
X_pca = pca.fit_transform(X)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_.cumsum()

plt.plot(explained_variance)

# apply the K-means on PCA
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 15, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X_pca)

plt.plot(y_kmeans, Y, 'ro')
plt.boxplot([Y[y_kmeans == 0], 
             Y[y_kmeans == 1],
             Y[y_kmeans == 2], 
             Y[y_kmeans == 3],
             Y[y_kmeans == 4], 
             Y[y_kmeans == 5]
             ])


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
history = autoencoder.fit(X1.values, Y1.values,
                epochs=1000,
                batch_size=200,
                shuffle=False,
                verbose = 2,
                validation_data=(X2.values, Y2.values))


plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# now let's evaluate the coding of the initial features
encoder = Model(input_layer, encoded)
preds = encoder.predict(X.values)

# apply the K-means on PCA
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(preds)

plt.plot(y_kmeans, Y, 'ro')
plt.boxplot([Y[y_kmeans == 0], 
             Y[y_kmeans == 1],
             Y[y_kmeans == 2], 
             Y[y_kmeans == 3],
             Y[y_kmeans == 4], 
             Y[y_kmeans == 5]
             ])

    
    
# KNN regrossor
from sklearn.neighbors import KNeighborsRegressor
regs = KNeighborsRegressor(n_neighbors = 100, metric='minkowski', p = 2)
regs.fit(X1, Y1)

#predicting the results
y_pred = regs.predict(X2)
print('r2 score', r2_score(Y2, y_pred )) 
