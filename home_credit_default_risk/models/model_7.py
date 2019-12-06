#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:45:03 2018

@author: neha

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
    f.write("model_7"+","+"1"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature and EXT_SOURCE_1 EXT_SOURCE_3 fillna mean"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

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

###########################################################
# 1.0 add feature income/annutiy to loan stress
###########################################################
application_train['income_annutiy_ratio']=application_train['AMT_ANNUITY']/application_train['AMT_INCOME_TOTAL']
important_columns.append('income_annutiy_ratio')

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
    f.write("model_7"+","+"1"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio income_annutiy_ratio feature and EXT_SOURCE_1 EXT_SOURCE_3 fillna mean"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)

"""
FLAG_OWN_CAR_N                                        1.044760
NAME_FAMILY_STATUS_Married                            1.056488
ORGANIZATION_TYPE_Self-employed                       1.092743
FLAG_DOCUMENT_3                                       1.137689
DAYS_LAST_PHONE_CHANGE                                1.170408
DEF_30_CNT_SOCIAL_CIRCLE                              1.265318
NAME_EDUCATION_TYPE_Higher education                  1.400846
income_annutiy_ratio                                  1.486174
REGION_RATING_CLIENT_W_CITY                           1.543733
NAME_INCOME_TYPE_Working                              1.571745
DAYS_ID_PUBLISH                                       2.324637
AMT_ANNUITY                                           3.242797
DAYS_EMPLOYED                                         3.711054
AMT_GOODS_PRICE                                       3.854398
CODE_GENDER_M                                         3.949211
DAYS_BIRTH                                            7.169715
EXT_SOURCE_1                                          9.334945
loan_annutiy_ratio                                   12.665753
EXT_SOURCE_2                                         13.380199
EXT_SOURCE_3                                         13.827097
"""


#################################################
# 3. PCA on address features
#################################################
apartment_columns = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
'YEARS_BUILD_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG',
'LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG',
'APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE',
'ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE',
'LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI',
'YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI',
'FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI',
'NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE',
'EMERGENCYSTATE_MODE']

apartment_object_columns = ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
pca = PCA(n_components=2)
apartment_non_object_columns = list(set(apartment_columns) - set(apartment_object_columns))
pc_dims = pca.fit_transform(application_train[apartment_non_object_columns].fillna(0))
print(pca.explained_variance_ratio_)   

application_train.loc[:, 'apartment_pc1'] = pc_dims[:, 0]
application_train.loc[:, 'apartment_pc2'] = pc_dims[:, 1]

important_columns.append('apartment_pc1')
important_columns.append('apartment_pc2')

common_columns = list(set(important_columns).intersection(apartment_columns))

important_columns = list(set(important_columns) - set(common_columns))

# run the model and get the results 
application_train['income_annutiy_ratio']=application_train['AMT_ANNUITY']/application_train['AMT_INCOME_TOTAL']
important_columns.append('income_annutiy_ratio')

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
    f.write("model_7"+","+"2"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio income_annutiy_ratio feature and EXT_SOURCE_1 EXT_SOURCE_3 fillna mean apartment pc"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)

"""
ORGANIZATION_TYPE_Self-employed                       1.026654
REG_CITY_NOT_LIVE_CITY                                1.073184
FLAG_OWN_CAR_N                                        1.169659
NAME_EDUCATION_TYPE_Secondary / secondary special     1.186938
FLAG_DOCUMENT_3                                       1.406827
apartment_pc1                                         1.609278
DAYS_LAST_PHONE_CHANGE                                1.615395
NAME_EDUCATION_TYPE_Higher education                  1.679077
NAME_INCOME_TYPE_Working                              1.803954
DEF_30_CNT_SOCIAL_CIRCLE                              1.806178
income_annutiy_ratio                                  1.999228
REGION_RATING_CLIENT_W_CITY                           2.014637
AMT_ANNUITY                                           2.634516
DAYS_ID_PUBLISH                                       2.654216
CODE_GENDER_M                                         3.321156
AMT_CREDIT                                            3.542976
DAYS_EMPLOYED                                         3.783510
AMT_GOODS_PRICE                                       6.002399
DAYS_BIRTH                                            7.626722
EXT_SOURCE_1                                         10.239593
EXT_SOURCE_2                                         13.312304
EXT_SOURCE_3                                         15.354604
"""
