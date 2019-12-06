#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:34:16 2018

@author: neha

Run on all columns and get important columsn again 

recalculate important columns
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import lightgbm as lgb
import random
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA



application_train = pd.read_csv('application_train.csv') # (307511, 122)
application_test = pd.read_csv('application_test.csv') # (48744, 121)

y_true_skid = list(application_train.loc[application_train['TARGET'] == 1, 'SK_ID_CURR'])
y_false_skid = list(application_train.loc[application_train['TARGET'] == 0, 'SK_ID_CURR'])

y_true = application_train[['SK_ID_CURR', 'TARGET']]
del application_train['TARGET']


mean_EXT_SOURCE_1 = application_train[~application_train.isnull()].EXT_SOURCE_1.mean()
mean_EXT_SOURCE_3 = application_train[~application_train.isnull()].EXT_SOURCE_3.mean()

application_train['EXT_SOURCE_1'] = application_train['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
application_train['EXT_SOURCE_3'] = application_train['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)


#################################################
# load train and dev set skids
#################################################
with open("train_all.txt", "rb") as fp:
    train_all_skid = pickle.load(fp)
    
with open("dev_all.txt", "rb") as fp:
    dev_all_skid = pickle.load(fp)
    
with open("dev_eyeball.txt", "rb") as fp:
    dev_eyeball_skid = pickle.load(fp)
    

###################################
# get False positive rate
###################################
def get_FPR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm[0][1]
    TN = cm[0][0]
    FPR = FP/(FP+TN)
    return FPR

#############################################
# one hot encoding
#############################################
x_object_columns = list(application_train.select_dtypes(['object']).columns)

one_hot_df = pd.concat([application_train,application_test])
one_hot_df = pd.get_dummies(one_hot_df, columns=x_object_columns)

application_train = one_hot_df.iloc[:application_train.shape[0],:]
application_test = one_hot_df.iloc[application_train.shape[0]:,]

#############################################
# 0.0 baseline model for this script
#############################################

application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']

x_columns = list(application_train.columns.drop(['SK_ID_CURR']))

X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), x_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), x_columns].fillna(0)
y_dev = y_true.loc[y_true.SK_ID_CURR.isin(dev_all_skid), 'TARGET']
scores_df = pd.Series()

###################################################
# train the model and get the training scores
###################################################

clf = GradientBoostingClassifier(n_estimators=500, random_state=0)
clf.fit(X_train, y_train)

# get training scores
y_train_predict = clf.predict(X_train)
y_train_predict_proba = clf.predict_proba(X_train)

scores_df['train_roc_auc'] = roc_auc_score(y_train, y_train_predict_proba[:, 1])
scores_df['train_accuracy'] = accuracy_score(y_train, y_train_predict)
scores_df['train_recall'] = recall_score(y_train, y_train_predict)
scores_df['train_fpr'] = get_FPR(y_train, y_train_predict)
scores_df['train_precision'] = precision_score(y_train, y_train_predict)
scores_df['train_f1'] = f1_score(y_train, y_train_predict)
         
###################################################
# results on the dev set
###################################################

y_dev_predict = clf.predict(X_dev)
y_dev_predict_proba = clf.predict_proba(X_dev)

# get dev scores
scores_df['test_roc_auc'] = roc_auc_score(y_dev, y_dev_predict_proba[:, 1])
scores_df['test_accuracy'] = accuracy_score(y_dev, y_dev_predict)
scores_df['test_recall'] = recall_score(y_dev, y_dev_predict)
scores_df['test_fpr'] = get_FPR(y_dev, y_dev_predict)
scores_df['test_precision'] = precision_score(y_dev, y_dev_predict)
scores_df['test_f1'] = f1_score(y_dev, y_dev_predict)


with open('all_reasults.csv', 'a') as f:
    f.write("model_8"+","+"0"+","+"GBC trained only on all columns of application data with dev set CV and loan_annutiy_ratio feature and EXT_SOURCE_1 EXT_SOURCE_3 fillna mean"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500 random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)

