#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:38:40 2018

@author: neha

EDA shows that quite a lot of data is Null
including a lot of important parameters
we need to check what happes if backfill with other than 0

Important features which are null 
OWN_CAR_AGE                     0.659908
EXT_SOURCE_1                    0.563811
AMT_REQ_CREDIT_BUREAU_YEAR      0.135016
AMT_REQ_CREDIT_BUREAU_QRT       0.135016
AMT_REQ_CREDIT_BUREAU_MON       0.135016
AMT_REQ_CREDIT_BUREAU_WEEK      0.135016
AMT_REQ_CREDIT_BUREAU_DAY       0.135016
AMT_REQ_CREDIT_BUREAU_HOUR      0.135016
EXT_SOURCE_3                    0.198253
OCCUPATION_TYPE                 0.313455

# Try fillna with mean, mode, median and from a distribution

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


application_train = pd.read_csv('application_train.csv') # (307511, 122)
application_test = pd.read_csv('application_test.csv') # (48744, 121)

y_true_skid = list(application_train.loc[application_train['TARGET'] == 1, 'SK_ID_CURR'])
y_false_skid = list(application_train.loc[application_train['TARGET'] == 0, 'SK_ID_CURR'])

y_true = application_train[['SK_ID_CURR', 'TARGET']]
del application_train['TARGET']

#################################################
# load train and dev set skids
#################################################
with open("train_all.txt", "rb") as fp:
    train_all_skid = pickle.load(fp)
    
with open("dev_all.txt", "rb") as fp:
    dev_all_skid = pickle.load(fp)
    
with open("dev_eyeball.txt", "rb") as fp:
    dev_eyeball_skid = pickle.load(fp)
    
###############################################
# from previous experiments, important columns
###############################################
important_columns = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
       'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
       'FLAG_WORK_PHONE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT_W_CITY',
       'HOUR_APPR_PROCESS_START', 'REG_CITY_NOT_LIVE_CITY', 'EXT_SOURCE_1',
       'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG',
       'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'FLOORSMAX_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
       'YEARS_BEGINEXPLUATATION_MODE', 'FLOORSMAX_MODE',
       'LIVINGAPARTMENTS_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
       'FLOORSMIN_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAREA_MEDI',
       'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
       'NAME_CONTRACT_TYPE_Cash loans', 'NAME_CONTRACT_TYPE_Revolving loans',
       'CODE_GENDER_F', 'CODE_GENDER_M', 'FLAG_OWN_CAR_N', 'FLAG_OWN_CAR_Y',
       'NAME_INCOME_TYPE_Commercial associate',
       'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Unemployed',
       'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Higher education',
       'NAME_EDUCATION_TYPE_Lower secondary',
       'NAME_EDUCATION_TYPE_Secondary / secondary special',
       'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married',
       'NAME_FAMILY_STATUS_Separated', 'NAME_HOUSING_TYPE_House / apartment',
       'NAME_HOUSING_TYPE_Municipal apartment',
       'NAME_HOUSING_TYPE_Rented apartment', 'OCCUPATION_TYPE_Core staff',
       'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_Laborers',
       'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Sales staff',
       'WEEKDAY_APPR_PROCESS_START_FRIDAY',
       'WEEKDAY_APPR_PROCESS_START_SATURDAY',
       'ORGANIZATION_TYPE_Business Entity Type 3',
       'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Industry: type 1',
       'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Realtor',
       'ORGANIZATION_TYPE_Self-employed', 'ORGANIZATION_TYPE_Trade: type 7',
       'ORGANIZATION_TYPE_Transport: type 3']

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
important_columns.append('loan_annutiy_ratio')

X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
y_dev = y_true.loc[y_true.SK_ID_CURR.isin(dev_all_skid), 'TARGET']
scores_df = pd.Series()

###################################################
# train the model and get the training scores
###################################################

clf = GradientBoostingClassifier(random_state=0)
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
    f.write("model_6"+","+"0"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)

"""
FLAG_OWN_CAR_Y                                        1.005291
NAME_EDUCATION_TYPE_Higher education                  1.243046
DAYS_LAST_PHONE_CHANGE                                1.369852
NAME_INCOME_TYPE_Working                              1.446174
DEF_30_CNT_SOCIAL_CIRCLE                              1.457484
REGION_RATING_CLIENT_W_CITY                           1.647758
AMT_CREDIT                                            2.379068
DAYS_ID_PUBLISH                                       2.432987
CODE_GENDER_M                                         2.690265
AMT_ANNUITY                                           3.085683
DAYS_EMPLOYED                                         3.611424
AMT_GOODS_PRICE                                       4.086893
EXT_SOURCE_1                                          6.944035
DAYS_BIRTH                                            7.730413
EXT_SOURCE_2                                         12.429327
loan_annutiy_ratio                                   12.493588
EXT_SOURCE_3                                         17.090728
"""


#############################################
# 1.0 fillna EXT_SOURCE_1 with mean
#############################################


application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']
important_columns.append('loan_annutiy_ratio')

mean_EXT_SOURCE_1 = application_train[~application_train.isnull()].EXT_SOURCE_1.mean()
application_train['EXT_SOURCE_1'] = application_train['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)


X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
y_dev = y_true.loc[y_true.SK_ID_CURR.isin(dev_all_skid), 'TARGET']
scores_df = pd.Series()

###################################################
# train the model and get the training scores
###################################################

clf = GradientBoostingClassifier(random_state=0)
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
    f.write("model_6"+","+"1"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature and EXT_SOURCE_1 fillna mean"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")


important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)


"""
FLAG_OWN_CAR_Y                                        1.065987
DAYS_LAST_PHONE_CHANGE                                1.272486
NAME_INCOME_TYPE_Working                              1.422555
NAME_EDUCATION_TYPE_Higher education                  1.467109
FLAG_DOCUMENT_3                                       1.483747
DEF_30_CNT_SOCIAL_CIRCLE                              1.545949
REGION_RATING_CLIENT_W_CITY                           1.607786
DAYS_ID_PUBLISH                                       1.898932
AMT_CREDIT                                            2.121918
CODE_GENDER_M                                         3.127048
AMT_GOODS_PRICE                                       3.458703
AMT_ANNUITY                                           3.507214
DAYS_EMPLOYED                                         3.797831
DAYS_BIRTH                                            6.278547
EXT_SOURCE_1                                          9.246583
EXT_SOURCE_2                                         12.307881
loan_annutiy_ratio                                   14.462997
EXT_SOURCE_3                                         16.557001
"""

#############################################
# 2.0 fillna EXT_SOURCE_3 with mean
#############################################

application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']
important_columns.append('loan_annutiy_ratio')

mean_EXT_SOURCE_1 = application_train[~application_train.isnull()].EXT_SOURCE_1.mean()
mean_EXT_SOURCE_3 = application_train[~application_train.isnull()].EXT_SOURCE_3.mean()

application_train['EXT_SOURCE_1'] = application_train['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
application_train['EXT_SOURCE_3'] = application_train['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)

X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
y_dev = y_true.loc[y_true.SK_ID_CURR.isin(dev_all_skid), 'TARGET']
scores_df = pd.Series()

###################################################
# train the model and get the training scores
###################################################

clf = GradientBoostingClassifier(random_state=0)
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
    f.write("model_6"+","+"2"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature and EXT_SOURCE_1, EXT_SOURCE_3 fillna mean"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")


important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)



##################################################################
# 3.0
# add more trees and submit
##################################################################
application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']
important_columns.append('loan_annutiy_ratio')

mean_EXT_SOURCE_1 = application_train[~application_train.isnull()].EXT_SOURCE_1.mean()
mean_EXT_SOURCE_3 = application_train[~application_train.isnull()].EXT_SOURCE_3.mean()

application_train['EXT_SOURCE_1'] = application_train['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
application_train['EXT_SOURCE_3'] = application_train['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)
application_test['EXT_SOURCE_1'] = application_test['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
application_test['EXT_SOURCE_3'] = application_test['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)


X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
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
    f.write("model_6"+","+"3"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature and EXT_SOURCE_1, EXT_SOURCE_3 fillna mean"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500  random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

#############################################
# refit the model and make prediction
#############################################
X = application_train[important_columns].fillna(0)
y = y_true['TARGET']
clf.fit(X, y)

application_test['loan_annutiy_ratio']=application_test['AMT_CREDIT']/application_test['AMT_ANNUITY']

X_test_predict = application_test[important_columns].fillna(0)
y_test_predict = clf.predict_proba(X_test_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
submission.to_csv('submission_7.csv', index=False)
                        

clf.feature_importances_
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)


"""
DEF_30_CNT_SOCIAL_CIRCLE                  1.009444
AMT_REQ_CREDIT_BUREAU_QRT                 1.095306
AMT_REQ_CREDIT_BUREAU_YEAR                1.155274
HOUR_APPR_PROCESS_START                   1.364096
REGION_POPULATION_RELATIVE                1.734324
AMT_CREDIT                                2.084237
DAYS_LAST_PHONE_CHANGE                    2.225380
AMT_INCOME_TOTAL                          2.330133
AMT_ANNUITY                               2.789989
DAYS_REGISTRATION                         2.963538
DAYS_ID_PUBLISH                           3.125645
DAYS_EMPLOYED                             3.321103
AMT_GOODS_PRICE                           3.570474
DAYS_BIRTH                                7.608921
EXT_SOURCE_1                              9.170220
EXT_SOURCE_3                              9.767919
EXT_SOURCE_2                             10.648014
loan_annutiy_ratio                       13.552888
"""

####################################################################
# 4.0
# Try for more fine grained fillna for EXT_SOURCE_1,EXT_SOURCE_3
# first fill by mean of OCCUPATION
# then fill by mean of NAME_INCOME_TYPE
# then fill by over all mean
####################################################################

application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']
application_test['loan_annutiy_ratio']=application_test['AMT_CREDIT']/application_test['AMT_ANNUITY']

important_columns.append('loan_annutiy_ratio')

mean_EXT_SOURCE_1_by_incometype = application_train.groupby('NAME_INCOME_TYPE').EXT_SOURCE_1.mean().reset_index()
mean_EXT_SOURCE_1_by_occupationtype = application_train.groupby('OCCUPATION_TYPE').EXT_SOURCE_1.mean().reset_index()

mean_EXT_SOURCE_3_by_incometype = application_train.groupby('NAME_INCOME_TYPE').EXT_SOURCE_3.mean().reset_index()
mean_EXT_SOURCE_3_by_occupationtype = application_train.groupby('OCCUPATION_TYPE').EXT_SOURCE_3.mean().reset_index()

mean_EXT_SOURCE_1 = application_train[~application_train.isnull()].EXT_SOURCE_1.mean()
mean_EXT_SOURCE_3 = application_train[~application_train.isnull()].EXT_SOURCE_3.mean()

application_train = pd.merge(application_train, mean_EXT_SOURCE_1_by_incometype, on='NAME_INCOME_TYPE', how='left', suffixes=('', '_income'))
application_train = pd.merge(application_train, mean_EXT_SOURCE_1_by_occupationtype, on='OCCUPATION_TYPE', how='left', suffixes=('', '_occupation'))

application_train = pd.merge(application_train, mean_EXT_SOURCE_3_by_incometype, on='NAME_INCOME_TYPE', how='left', suffixes=('', '_income'))
application_train = pd.merge(application_train, mean_EXT_SOURCE_3_by_occupationtype, on='OCCUPATION_TYPE', how='left', suffixes=('', '_occupation'))


application_train.loc[application_train.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'] = application_train.loc[application_train.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1_occupation']
application_train.loc[application_train.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'] = application_train.loc[application_train.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1_income']
application_train.loc[application_train.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'] = application_train.loc[application_train.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3_occupation']
application_train.loc[application_train.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'] = application_train.loc[application_train.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3_income']
application_train.loc[application_train.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'] = application_train.loc[application_train.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
application_train.loc[application_train.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'] = application_train.loc[application_train.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)


application_test = pd.merge(application_test, mean_EXT_SOURCE_1_by_incometype, on='NAME_INCOME_TYPE', how='left', suffixes=('', '_income'))
application_test = pd.merge(application_test, mean_EXT_SOURCE_1_by_occupationtype, on='OCCUPATION_TYPE', how='left', suffixes=('', '_occupation'))

application_test = pd.merge(application_test, mean_EXT_SOURCE_3_by_incometype, on='NAME_INCOME_TYPE', how='left', suffixes=('', '_income'))
application_test = pd.merge(application_test, mean_EXT_SOURCE_3_by_occupationtype, on='OCCUPATION_TYPE', how='left', suffixes=('', '_occupation'))


application_test.loc[application_test.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'] = application_test.loc[application_test.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1_occupation']
application_test.loc[application_test.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'] = application_test.loc[application_test.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1_income']
application_test.loc[application_test.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'] = application_test.loc[application_test.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3_occupation']
application_test.loc[application_test.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'] = application_test.loc[application_test.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3_income']
application_test.loc[application_test.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'] = application_test.loc[application_test.EXT_SOURCE_1.isnull(), 'EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
application_test.loc[application_test.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'] = application_test.loc[application_test.EXT_SOURCE_3.isnull(), 'EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)


#############################################
# one hot encoding
#############################################
x_object_columns = list(application_train.select_dtypes(['object']).columns)

one_hot_df = pd.concat([application_train,application_test])
one_hot_df = pd.get_dummies(one_hot_df, columns=x_object_columns)

application_train = one_hot_df.iloc[:application_train.shape[0],:]
application_test = one_hot_df.iloc[application_train.shape[0]:,]

############################################
# Split the train and dev set
############################################

X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
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
    f.write("model_6"+","+"4"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature and EXT_SOURCE_1 EXT_SOURCE_3 fillna by group"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500  random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

#############################################
# refit the model and make prediction
#############################################
X = application_train[important_columns].fillna(0)
y = y_true['TARGET']
clf.fit(X, y)

X_test_predict = application_test[important_columns].fillna(0)
y_test_predict = clf.predict_proba(X_test_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
submission.to_csv('submission_8.csv', index=False)
                        
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

"""
OWN_CAR_AGE                                  1.012735
CODE_GENDER_M                                1.077155
AMT_REQ_CREDIT_BUREAU_QRT                    1.314297
AMT_REQ_CREDIT_BUREAU_YEAR                   1.348966
HOUR_APPR_PROCESS_START                      1.519758
AMT_CREDIT                                   1.940590
REGION_POPULATION_RELATIVE                   2.015375
DAYS_LAST_PHONE_CHANGE                       2.359402
DAYS_REGISTRATION                            2.449548
AMT_INCOME_TOTAL                             2.482419
DAYS_ID_PUBLISH                              2.898912
AMT_GOODS_PRICE                              2.964009
AMT_ANNUITY                                  3.180730
DAYS_EMPLOYED                                3.201080
DAYS_BIRTH                                   6.398361
EXT_SOURCE_1                                10.248929
EXT_SOURCE_3                                10.475073
EXT_SOURCE_2                                10.793852
loan_annutiy_ratio                          13.681443
"""