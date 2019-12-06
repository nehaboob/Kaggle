#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:14:49 2018

@author: neha

try adding other data sources
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
from lightgbm import LGBMClassifier

application_train = pd.read_csv('application_train.csv') # (307511, 122)
application_test = pd.read_csv('application_test.csv') # (48744, 121)


y_true_skid = list(application_train.loc[application_train['TARGET'] == 1, 'SK_ID_CURR'])
y_false_skid = list(application_train.loc[application_train['TARGET'] == 0, 'SK_ID_CURR'])

y_true = application_train[['SK_ID_CURR', 'TARGET']]
del application_train['TARGET']

application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']

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
       'ORGANIZATION_TYPE_Transport: type 3', 'loan_annutiy_ratio']

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
    f.write("model_9"+","+"0"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature and EXT_SOURCE_1 EXT_SOURCE_3 fillna mean"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)

"""
FLAG_DOCUMENT_3                                       1.138692
ORGANIZATION_TYPE_Self-employed                       1.177906
DAYS_LAST_PHONE_CHANGE                                1.339549
DEF_30_CNT_SOCIAL_CIRCLE                              1.431089
NAME_INCOME_TYPE_Working                              1.545842
NAME_EDUCATION_TYPE_Higher education                  1.554133
REGION_RATING_CLIENT_W_CITY                           1.776740
AMT_CREDIT                                            1.915012
DAYS_ID_PUBLISH                                       2.209778
DAYS_EMPLOYED                                         3.537986
CODE_GENDER_M                                         3.649247
AMT_ANNUITY                                           3.826076
AMT_GOODS_PRICE                                       3.965517
DAYS_BIRTH                                            6.643630
EXT_SOURCE_1                                          9.140697
loan_annutiy_ratio                                   11.503518
EXT_SOURCE_2                                         13.348712
EXT_SOURCE_3                                         13.649269
"""

##################################################################
# try adding bureau.csv features
# number of active total loans
##################################################################
bureau = pd.read_csv('bureau.csv')

#bureau.info()
#bureau.CREDIT_ACTIVE.unique()
#bureau.CREDIT_ACTIVE.value_counts()

bureau_loan_total = bureau.groupby('SK_ID_CURR').size().reset_index(name='total_loans')
bureau_loan_status_total = bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE']).size().reset_index(name='total_loans')
bureau_loan_status_total = bureau_loan_status_total.pivot(index='SK_ID_CURR', columns='CREDIT_ACTIVE', values='total_loans').fillna(0).reset_index()


application_train = pd.merge(application_train, bureau_loan_total, on='SK_ID_CURR', how='left')
application_train = pd.merge(application_train, bureau_loan_status_total, on='SK_ID_CURR', how='left')

important_columns.extend(['total_loans', 'Active', 'Bad debt', 'Closed', 'Sold'])


###################################################    
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
    f.write("model_9"+","+"1"+","+"GBC trained only on important columns of application data with dev set CV adding bureau features"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)


"""
FLAG_OWN_CAR_Y                                        1.122437
FLAG_DOCUMENT_3                                       1.160844
DAYS_LAST_PHONE_CHANGE                                1.182193
DEF_30_CNT_SOCIAL_CIRCLE                              1.329769
NAME_EDUCATION_TYPE_Higher education                  1.500847
NAME_INCOME_TYPE_Working                              1.547390
REGION_RATING_CLIENT_W_CITY                           1.716783
DAYS_ID_PUBLISH                                       1.820790
AMT_CREDIT                                            2.227402
Closed                                                2.391737
CODE_GENDER_M                                         2.510946
AMT_ANNUITY                                           3.077497
DAYS_EMPLOYED                                         3.463190
Active                                                3.812619
AMT_GOODS_PRICE                                       3.970192
DAYS_BIRTH                                            6.255657
EXT_SOURCE_1                                          9.120606
EXT_SOURCE_3                                         12.034292
loan_annutiy_ratio                                   12.738926
EXT_SOURCE_2                                         13.204072
"""

#########################################################
# add more features 
#########################################################

#bureau.info()
#bureau.CREDIT_ACTIVE.unique()
#bureau.CREDIT_ACTIVE.value_counts()

bureau_latest_loan = bureau.groupby('SK_ID_CURR').DAYS_CREDIT.max().reset_index()
bureau_max_overdue = bureau.groupby('SK_ID_CURR').CREDIT_DAY_OVERDUE.max().reset_index()

application_train = pd.merge(application_train, bureau_latest_loan, on='SK_ID_CURR', how='left')
application_train = pd.merge(application_train, bureau_max_overdue, on='SK_ID_CURR', how='left')

important_columns.extend(['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE'])


###################################################    
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
    f.write("model_9"+","+"2"+","+"GBC trained only on important columns of application data with dev set CV adding bureau features"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)


"""
DAYS_LAST_PHONE_CHANGE                                1.006218
FLAG_OWN_CAR_Y                                        1.067647
ORGANIZATION_TYPE_Self-employed                       1.152420
FLAG_DOCUMENT_3                                       1.165599
DEF_30_CNT_SOCIAL_CIRCLE                              1.366505
REGION_RATING_CLIENT_W_CITY                           1.433676
NAME_INCOME_TYPE_Working                              1.505599
NAME_EDUCATION_TYPE_Higher education                  1.507062
Closed                                                2.114597
DAYS_ID_PUBLISH                                       2.158017
AMT_ANNUITY                                           2.375726
DAYS_CREDIT                                           2.570540
AMT_CREDIT                                            2.603369
CODE_GENDER_M                                         3.320955
DAYS_EMPLOYED                                         3.340673
AMT_GOODS_PRICE                                       3.424828
Active                                                4.018866
DAYS_BIRTH                                            6.611745
EXT_SOURCE_1                                          8.885746
loan_annutiy_ratio                                   11.274952
EXT_SOURCE_3                                         11.909292
EXT_SOURCE_2                                         12.457694
"""

#####################################################################
# add more features
#####################################################################

bureau_DAYS_CREDIT_ENDDATE_sum = bureau.groupby('SK_ID_CURR').DAYS_CREDIT_ENDDATE.sum().reset_index(name='bureau_DAYS_CREDIT_ENDDATE_sum')
bureau_DAYS_CREDIT_ENDDATE_min = bureau.groupby('SK_ID_CURR').DAYS_CREDIT_ENDDATE.min().reset_index(name='bureau_DAYS_CREDIT_ENDDATE_min')
bureau_DAYS_CREDIT_ENDDATE_max = bureau.groupby('SK_ID_CURR').DAYS_CREDIT_ENDDATE.max().reset_index(name='bureau_DAYS_CREDIT_ENDDATE_max')

bureau_DAYS_ENDDATE_FACT_sum = bureau.groupby('SK_ID_CURR').DAYS_ENDDATE_FACT.sum().reset_index(name='bureau_DAYS_ENDDATE_FACT_sum')
bureau_DAYS_ENDDATE_FACT_min = bureau.groupby('SK_ID_CURR').DAYS_ENDDATE_FACT.min().reset_index(name='bureau_DAYS_ENDDATE_FACT_min')
bureau_DAYS_ENDDATE_FACT_max = bureau.groupby('SK_ID_CURR').DAYS_ENDDATE_FACT.max().reset_index(name='bureau_DAYS_ENDDATE_FACT_max')


application_train = pd.merge(application_train, bureau_DAYS_CREDIT_ENDDATE_sum, on='SK_ID_CURR', how='left')
application_train = pd.merge(application_train, bureau_DAYS_CREDIT_ENDDATE_min, on='SK_ID_CURR', how='left')
application_train = pd.merge(application_train, bureau_DAYS_CREDIT_ENDDATE_max, on='SK_ID_CURR', how='left')

application_train = pd.merge(application_train, bureau_DAYS_ENDDATE_FACT_sum, on='SK_ID_CURR', how='left')
application_train = pd.merge(application_train, bureau_DAYS_ENDDATE_FACT_min, on='SK_ID_CURR', how='left')
application_train = pd.merge(application_train, bureau_DAYS_ENDDATE_FACT_max, on='SK_ID_CURR', how='left')

important_columns.extend(['bureau_DAYS_CREDIT_ENDDATE_sum', 'bureau_DAYS_CREDIT_ENDDATE_min', 'bureau_DAYS_CREDIT_ENDDATE_max',
'bureau_DAYS_ENDDATE_FACT_sum', 'bureau_DAYS_ENDDATE_FACT_min', 'bureau_DAYS_ENDDATE_FACT_max'])

###################################################    
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
    f.write("model_9"+","+"3"+","+"GBC trained only on important columns of application data with dev set CV adding bureau features"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)


"""

NAME_INCOME_TYPE_Working                              1.024178
NAME_FAMILY_STATUS_Married                            1.036943
FLAG_OWN_CAR_N                                        1.232856
bureau_DAYS_CREDIT_ENDDATE_sum                        1.292811
NAME_EDUCATION_TYPE_Higher education                  1.360490
FLAG_DOCUMENT_3                                       1.481961
DEF_30_CNT_SOCIAL_CIRCLE                              1.497594
REGION_RATING_CLIENT_W_CITY                           1.562036
DAYS_ID_PUBLISH                                       1.833012
AMT_CREDIT                                            1.874565
AMT_ANNUITY                                           2.633519
DAYS_CREDIT                                           2.786284
CODE_GENDER_M                                         2.822985
DAYS_EMPLOYED                                         3.176877
Active                                                3.277507
AMT_GOODS_PRICE                                       3.575015
DAYS_BIRTH                                            6.275797
EXT_SOURCE_1                                          8.787345
loan_annutiy_ratio                                   11.491193
EXT_SOURCE_3                                         11.548941
EXT_SOURCE_2                                         12.370567
"""


#####################################################################
# add more features
#####################################################################
bureau_agg_columns = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY'].sum().reset_index()

application_train = pd.merge(application_train, bureau_agg_columns, on='SK_ID_CURR', how='left')

important_columns.extend(['AMT_CREDIT_MAX_OVERDUE',
       'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
       'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY_y'])
important_columns.append('AMT_ANNUITY_x')
important_columns.remove('AMT_ANNUITY')    
    
###################################################    
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
    f.write("model_9"+","+"4"+","+"GBC trained only on important columns of application data with dev set CV adding bureau features"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)


"""
DAYS_LAST_PHONE_CHANGE                                1.003796
bureau_DAYS_ENDDATE_FACT_sum                          1.238264
DEF_30_CNT_SOCIAL_CIRCLE                              1.241874
FLAG_OWN_CAR_N                                        1.253135
FLAG_DOCUMENT_3                                       1.287723
bureau_DAYS_CREDIT_ENDDATE_sum                        1.369659
AMT_CREDIT_SUM_DEBT                                   1.382392
REGION_RATING_CLIENT_W_CITY                           1.397330
AMT_CREDIT_MAX_OVERDUE                                1.413882
AMT_ANNUITY_x                                         1.479911
DAYS_ID_PUBLISH                                       1.493160
NAME_INCOME_TYPE_Working                              1.514753
NAME_EDUCATION_TYPE_Higher education                  1.556094
AMT_ANNUITY_x                                         2.045919
DAYS_CREDIT                                           2.146512
AMT_CREDIT_SUM_OVERDUE                                2.482743
CODE_GENDER_M                                         2.635913
Active                                                2.843257
AMT_GOODS_PRICE                                       3.150678
DAYS_EMPLOYED                                         3.182703
DAYS_BIRTH                                            5.715415
EXT_SOURCE_1                                          8.344862
loan_annutiy_ratio                                   10.379503
EXT_SOURCE_3                                         11.146219
EXT_SOURCE_2                                         11.958161
"""

##############################
# add more trees
##############################
###################################################    
X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
y_dev = y_true.loc[y_true.SK_ID_CURR.isin(dev_all_skid), 'TARGET']
scores_df = pd.Series()

###################################################
# train the model and get the training scores
###################################################

clf = GradientBoostingClassifier(n_estimators=500 , random_state=0)
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
    f.write("model_9"+","+"5"+","+"GBC trained only on important columns of application data with dev set CV adding bureau features"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500 random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)


"""
bureau_DAYS_ENDDATE_FACT_max             1.050284
HOUR_APPR_PROCESS_START                  1.080204
AMT_ANNUITY_x                            1.105995
AMT_ANNUITY_y                            1.218676
AMT_ANNUITY_x                            1.348352
DAYS_LAST_PHONE_CHANGE                   1.473796
REGION_POPULATION_RELATIVE               1.513800
DAYS_REGISTRATION                        1.593553
Active                                   1.695175
AMT_GOODS_PRICE                          1.747933
AMT_CREDIT_MAX_OVERDUE                   1.823975
AMT_INCOME_TOTAL                         1.860005
DAYS_CREDIT                              1.913746
AMT_CREDIT                               2.017171
AMT_CREDIT_SUM_LIMIT                     2.020402
DAYS_EMPLOYED                            2.053736
AMT_CREDIT_SUM_DEBT                      2.078540
AMT_CREDIT_SUM_OVERDUE                   2.168668
bureau_DAYS_CREDIT_ENDDATE_min           2.186319
DAYS_ID_PUBLISH                          2.310000
AMT_CREDIT_SUM                           2.541077
bureau_DAYS_CREDIT_ENDDATE_sum           2.776944
bureau_DAYS_CREDIT_ENDDATE_max           2.927957
DAYS_BIRTH                               5.516040
EXT_SOURCE_1                             6.907416
EXT_SOURCE_2                             7.782941
EXT_SOURCE_3                             8.207163
loan_annutiy_ratio                       8.917022
"""

###############################
# prepare the submission file
##############################
application_test['loan_annutiy_ratio']=application_test['AMT_CREDIT']/application_test['AMT_ANNUITY']
application_test['EXT_SOURCE_1'] = application_test['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
application_test['EXT_SOURCE_3'] = application_test['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)

application_test = pd.merge(application_test, bureau_loan_total, on='SK_ID_CURR', how='left')
application_test = pd.merge(application_test, bureau_loan_status_total, on='SK_ID_CURR', how='left')

application_test = pd.merge(application_test, bureau_latest_loan, on='SK_ID_CURR', how='left')
application_test = pd.merge(application_test, bureau_max_overdue, on='SK_ID_CURR', how='left')

application_test = pd.merge(application_test, bureau_DAYS_CREDIT_ENDDATE_sum, on='SK_ID_CURR', how='left')
application_test = pd.merge(application_test, bureau_DAYS_CREDIT_ENDDATE_min, on='SK_ID_CURR', how='left')
application_test = pd.merge(application_test, bureau_DAYS_CREDIT_ENDDATE_max, on='SK_ID_CURR', how='left')

application_test = pd.merge(application_test, bureau_DAYS_ENDDATE_FACT_sum, on='SK_ID_CURR', how='left')
application_test = pd.merge(application_test, bureau_DAYS_ENDDATE_FACT_min, on='SK_ID_CURR', how='left')
application_test = pd.merge(application_test, bureau_DAYS_ENDDATE_FACT_max, on='SK_ID_CURR', how='left')

application_test = pd.merge(application_test, bureau_agg_columns, on='SK_ID_CURR', how='left')

#############################################
# refit the model and make prediction
#############################################
X = application_train[important_columns].fillna(0)
y = y_true['TARGET']
clf.fit(X, y)

X_test_predict = application_test[important_columns].fillna(0)
y_test_predict = clf.predict_proba(X_test_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
submission.to_csv('submission_9.csv', index=False)
                        
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

"""
REGION_POPULATION_RELATIVE                1.052001
bureau_DAYS_ENDDATE_FACT_sum              1.103469
bureau_DAYS_ENDDATE_FACT_max              1.131317
AMT_ANNUITY_y                             1.161648
AMT_ANNUITY_x                             1.274226
AMT_ANNUITY_x                             1.308887
DAYS_LAST_PHONE_CHANGE                    1.454606
AMT_CREDIT_SUM_LIMIT                      1.623228
AMT_CREDIT_MAX_OVERDUE                    1.631812
AMT_CREDIT                                1.754160
Active                                    1.800866
DAYS_REGISTRATION                         1.879046
DAYS_ID_PUBLISH                           1.936457
DAYS_CREDIT                               2.106306
AMT_CREDIT_SUM_OVERDUE                    2.164133
bureau_DAYS_CREDIT_ENDDATE_min            2.176531
AMT_CREDIT_SUM_DEBT                       2.225585
AMT_CREDIT_SUM                            2.265737
AMT_GOODS_PRICE                           2.308281
DAYS_EMPLOYED                             2.409363
bureau_DAYS_CREDIT_ENDDATE_max            2.561073
bureau_DAYS_CREDIT_ENDDATE_sum            2.914310
DAYS_BIRTH                                5.057740
EXT_SOURCE_1                              6.414758
EXT_SOURCE_3                              7.633590
EXT_SOURCE_2                              7.930822
loan_annutiy_ratio                       10.078785
"""

#######################################
# check lightGBM model
#######################################
X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
y_dev = y_true.loc[y_true.SK_ID_CURR.isin(dev_all_skid), 'TARGET']

clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_dev, y_dev)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

# get training scores
y_train_predict_proba = clf.predict_proba(X_train, num_iteration=clf.best_iteration)
y_train_predict = (y_train_predict > 0.5)*1 

scores_df['train_roc_auc'] = roc_auc_score(y_train, y_train_predict_proba[:, 1])
scores_df['train_accuracy'] = accuracy_score(y_train, y_train_predict)
scores_df['train_recall'] = recall_score(y_train, y_train_predict)
scores_df['train_fpr'] = get_FPR(y_train, y_train_predict)
scores_df['train_precision'] = precision_score(y_train, y_train_predict)
scores_df['train_f1'] = f1_score(y_train, y_train_predict)
         
###################################################
# results on the dev set
###################################################

y_dev_predict_proba = clf.predict_proba(X_dev, num_iteration=clf.best_iteration)
y_dev_predict = (y_dev_predict > 0.5)*1

# get dev scores
scores_df['test_roc_auc'] = roc_auc_score(y_dev, y_dev_predict_proba[:, 1])
scores_df['test_accuracy'] = accuracy_score(y_dev, y_dev_predict)
scores_df['test_recall'] = recall_score(y_dev, y_dev_predict)
scores_df['test_fpr'] = get_FPR(y_dev, y_dev_predict)
scores_df['test_precision'] = precision_score(y_dev, y_dev_predict)
scores_df['test_f1'] = f1_score(y_dev, y_dev_predict)


with open('all_reasults.csv', 'a') as f:
    f.write("model_9"+","+"6"+","+"LightGBM trained only on important columns of application data with dev set CV adding bureau features"+","+"LGBMClassifier"+","+"LGBMClassifier"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")


important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)


########################################
# predict
########################################

X_test_predict = application_test[important_columns].fillna(0)
y_test_predict = clf.predict_proba(X_test_predict, num_iteration=clf.best_iteration)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
submission.to_csv('submission_10.csv', index=False)
                        

