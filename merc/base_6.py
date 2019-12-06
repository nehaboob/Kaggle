#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 12:48:52 2017

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

def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    #print('MSE: %2.3f' % mse)
    return mse

def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     #print('R2: %2.3f' % r2)
     return r2

def two_score(y_true,y_pred):    
    MSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    score = R2(y_true,y_pred)
    return score

def two_scorer():
    return make_scorer(two_score, greater_is_better=True) # change for false if using MSE

# exp1 -- with outlier XGB
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


val = pd.DataFrame()
val['y_val'] = y_val
val['y_pred'] = optimized_GBM.best_estimator_.predict(x_val)


# exp2 -- without outlier XGB
cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(x_out_tr, y_out_tr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

val_out = pd.DataFrame()
val_out['y_out_val'] = y_val
val_out['y_out_pred'] = optimized_GBM.best_estimator_.predict(x_val)



# ecp 3 apply PCA - with outlier

from sklearn.decomposition import PCA
pca = PCA(n_components = 150)
x_pca_tr = pca.fit_transform(x_tr)
x_pca_val = pca.transform(x_val)
explained_variance = pca.explained_variance_ratio_.cumsum()

cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(x_pca_tr, y_tr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_pca_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

val['y_pca_val'] = y_val
val['y_pca_pred'] = optimized_GBM.best_estimator_.predict(x_pca_val)



# exp 4 pCA t without outlier

from sklearn.decomposition import PCA
pca = PCA(n_components = 150)
x_pca_out_tr = pca.fit_transform(x_out_tr)
x_pca_out_val = pca.transform(x_val)
explained_variance = pca.explained_variance_ratio_.cumsum()

cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(x_pca_out_tr, y_out_tr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_pca_out_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

val_out['y_pca_out_val'] = y_val
val_out['y_pca_out_pred'] = optimized_GBM.best_estimator_.predict(x_pca_out_val)

val.to_csv('xgb_val.csv', index=False)
val_out.to_csv('xgb_val_out.csv', index=False)


plt.scatter(range(0, 783), val['y_pred'], color='red')
plt.scatter(range(0, 783), val_out['y_out_pred'], color='green')
plt.scatter(range(0, 783), y_val, color='blue')



#exp 5 autoencoder with outlier
from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 50
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

# let's create and compile the autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# these parameters seems to work for the Mercedes dataset
history = autoencoder.fit(x_tr.values, x_tr.values,
                epochs=500,
                batch_size=200,
                shuffle=False,
                verbose = 2,
                validation_data=None)


plt.plot(history.history['loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# now let's evaluate the coding of the initial features
encoder = Model(input_layer, encoded)
x_en_tr = encoder.predict(x_tr.values)
x_en_val = encoder.predict(x_val.values)

cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(x_en_tr, y_tr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_en_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)



#exp 6 autoencoder without outlier
from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 50
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

# let's create and compile the autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# these parameters seems to work for the Mercedes dataset
history = autoencoder.fit(x_out_tr.values, x_out_tr.values,
                epochs=500,
                batch_size=200,
                shuffle=False,
                verbose = 2,
                validation_data=None)


plt.plot(history.history['loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# now let's evaluate the coding of the initial features
encoder = Model(input_layer, encoded)
x_en_tr = encoder.predict(x_out_tr.values)
x_en_val = encoder.predict(x_val.values)

cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(x_en_tr, y_out_tr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_en_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)




## with cluster experiments


# apply the K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 20, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x_tr)
y_val_kmeans = kmeans.predict(x_val)

x_k_tr= x_tr.copy()
x_k_val = x_val.copy()
x_k_tr = np.column_stack((x_k_tr, y_kmeans))
x_k_val = np.column_stack((x_k_val, y_val_kmeans))
     
cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(x_k_tr, y_tr)

print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_k_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)


val['y_k_val'] = y_val
val['y_k_pred'] = optimized_GBM.best_estimator_.predict(x_k_val)

plt.hist(optimized_GBM.best_estimator_.predict(x_k_val), 200)

reg = optimized_GBM.best_estimator_.booster().get_fscore()

#apply logic to seclet ycval
out_val = val['y_pred'][val['y_pred']  > 110].index.values  

f_val = val_out['y_out_pred'] 
f_val[out_val] = val['y_pred'][out_val]


val['f_val'] = f_val
   
print('r2 score', r2_score(y_val, f_val)) 

second = np.column_stack((val['y_pred'], val_out['y_out_pred']))
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(second, y_val)

print('r2 score', r2_score(y_val, regressor.predict(second))) 


from sklearn.ensemble import RandomForestRegressor
regressor_r = RandomForestRegressor(n_estimators = 10, criterion = 'entropy', random_state = 0)
regressor_r.fit(second, y_val)
#predicting the results

## final submission
# exp1 -- with outlier XGB
cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model = xgb.XGBRegressor(nthread=4)
 
optimized_GBM = GridSearchCV(xgb_model, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM.fit(X, Y)
y_test_1 = optimized_GBM.best_estimator_.predict(X_test)


print('r2 score', r2_score(y_val, optimized_GBM.best_estimator_.predict(x_val))) 
print('CV score', optimized_GBM.best_score_)
print('best params', optimized_GBM.best_params_)

# exp2 -- with outlier XGB
cv_params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,5)],  'subsample':[i/10.0 for i in range(8,11)],
'colsample_bytree':[i/10.0 for i in range(8,11)], 'max_depth': [2,3], 'n_estimators': [100]}

xgb_model_2 = xgb.XGBRegressor(nthread=4)
 
optimized_GBM_2 = GridSearchCV(xgb_model_2, 
                             cv_params, 
                             scoring = two_scorer(), verbose = 1) 

optimized_GBM_2.fit(X_out, Y_out)
y_test_2 = optimized_GBM_2.best_estimator_.predict(X_test)


second_1 = np.column_stack((y_test_1, y_test_2))


submit = regressor.predict(second_1)

sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = submit
sub.to_csv('xgb_second.csv', index=False)


## avg
sub = pd.DataFrame()
sub['ID'] = test["ID"]
sub['y'] = (y_test_1+y_test_2)/2
sub.to_csv('xgb_avg.csv', index=False)


#apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
x_pca = pca.fit_transform(X)

plt.scatter(x_pca, Y)

var = train.X232
plt.hist(Y[var==0], 200, alpha=0.5, label='0', color = plt.cm.jet(0))
plt.hist(Y[var==1], 200, alpha=0.75, label='1', color = plt.cm.jet(255))
plt.title('X232')
