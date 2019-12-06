#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:56:50 2018

@author: neha

we will try feature engineering in this file

"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, confusion_matrix
import lightgbm as lgb

application_train = pd.read_csv('application_train.csv') # (307511, 122)
application_test = pd.read_csv('application_test.csv') # (48744, 121)

y_true = application_train['TARGET']
del application_train['TARGET']

def get_FPR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm[0][1]
    TN = cm[0][0]
    FPR = FP/(FP+TN)
    return FPR

scoring = {'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall', 
           'f1':'f1',
           'fpr': make_scorer(get_FPR),
           'roc_auc': 'roc_auc'
           }

# from previous experiments
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

# one hot encoding

x_object_columns = list(application_train.select_dtypes(['object']).columns)

one_hot_df = pd.concat([application_train,application_test])
one_hot_df = pd.get_dummies(one_hot_df, columns=x_object_columns)

application_train = one_hot_df.iloc[:application_train.shape[0],:]
application_test = one_hot_df.iloc[application_train.shape[0]:,]


##################################
# baseline model for this script
##################################
X = application_train[important_columns].fillna(0)
y = y_true

clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()


with open('all_reasults.csv', 'a') as f:
    f.write("model_2"+","+"0"+","+"GBC trained only on important columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

"""
OWN_CAR_AGE                                           1.066155
DAYS_REGISTRATION                                     1.088814
NAME_INCOME_TYPE_Working                              1.121366
NAME_FAMILY_STATUS_Married                            1.252210
FLAG_OWN_CAR_Y                                        1.275174
FLAG_DOCUMENT_3                                       1.373327
DAYS_LAST_PHONE_CHANGE                                1.523044
REGION_RATING_CLIENT_W_CITY                           1.573710
NAME_EDUCATION_TYPE_Higher education                  1.653849
DEF_30_CNT_SOCIAL_CIRCLE                              1.773310
CODE_GENDER_M                                         2.281464
DAYS_ID_PUBLISH                                       3.301761
AMT_CREDIT                                            3.742677
AMT_ANNUITY                                           3.954930
DAYS_EMPLOYED                                         4.338528
EXT_SOURCE_1                                          5.695660
AMT_GOODS_PRICE                                       5.843150
DAYS_BIRTH                                            7.564957
EXT_SOURCE_2                                         14.823958
EXT_SOURCE_3                                         18.014041
"""

#########################################
# add feature loan and income ratio
#########################################
application_train['loan_income_ratio']=application_train['AMT_CREDIT']/application_train['AMT_INCOME_TOTAL']
application_test['loan_income_ratio']=application_test['AMT_CREDIT']/application_test['AMT_INCOME_TOTAL']

important_columns.append('loan_income_ratio')

X = application_train[important_columns].fillna(0)
y = y_true

clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_2"+","+"1"+","+"GBC trained only on important columns of application data and feature loan_income_ratio"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

"""
NAME_FAMILY_STATUS_Married                            1.103397
FLAG_OWN_CAR_Y                                        1.399871
REGION_RATING_CLIENT_W_CITY                           1.405661
FLAG_DOCUMENT_3                                       1.494260
DAYS_LAST_PHONE_CHANGE                                1.505681
NAME_INCOME_TYPE_Working                              1.573865
NAME_EDUCATION_TYPE_Higher education                  1.659993
DEF_30_CNT_SOCIAL_CIRCLE                              1.679984
loan_income_ratio                                     1.787273
CODE_GENDER_M                                         2.840316
DAYS_ID_PUBLISH                                       2.942779
AMT_ANNUITY                                           3.086323
DAYS_EMPLOYED                                         4.147954
AMT_CREDIT                                            4.199062
EXT_SOURCE_1                                          5.824336
AMT_GOODS_PRICE                                       6.242495
DAYS_BIRTH                                            8.084682
EXT_SOURCE_2                                         13.398163
EXT_SOURCE_3                                         19.693420
"""


############
# add loan_annutiy_ratio as feature
############
application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']
application_test['loan_annutiy_ratio']=application_test['AMT_CREDIT']/application_test['AMT_ANNUITY']

important_columns.append('loan_annutiy_ratio')
                                          
X = application_train[important_columns].fillna(0)
y = y_true

clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_2"+","+"2"+","+"GBC trained only on important columns of application data and feature added loan_annutiy_ratio"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

"""
loan_income_ratio                                     1.008351
NAME_FAMILY_STATUS_Married                            1.052098
DAYS_LAST_PHONE_CHANGE                                1.208083
FLAG_DOCUMENT_3                                       1.219718
AMT_CREDIT                                            1.307857
FLAG_OWN_CAR_Y                                        1.309377
NAME_INCOME_TYPE_Working                              1.348661
NAME_EDUCATION_TYPE_Higher education                  1.406713
REGION_RATING_CLIENT_W_CITY                           1.487430
DEF_30_CNT_SOCIAL_CIRCLE                              1.723827
DAYS_ID_PUBLISH                                       2.589210
CODE_GENDER_M                                         2.886930
AMT_ANNUITY                                           3.340116
DAYS_EMPLOYED                                         3.871947
EXT_SOURCE_1                                          5.177946
AMT_GOODS_PRICE                                       5.600747
DAYS_BIRTH                                            6.913109
EXT_SOURCE_2                                         13.022441
loan_annutiy_ratio                                   14.071362
EXT_SOURCE_3                                         16.387901
"""

############
# add loan_days_employed_ratio as feature
############
application_train['days_employed_loan_ratio']=application_train['DAYS_EMPLOYED']/application_train['AMT_CREDIT']
application_test['days_employed_loan_ratio']=application_test['DAYS_EMPLOYED']/application_test['AMT_CREDIT']

important_columns.append('days_employed_loan_ratio')

X = application_train[important_columns].fillna(0)
y = y_true

clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_2"+","+"3"+","+"GBC trained only on important columns of application data and feature added loan_days_employed_ratio"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)


"""
FLAG_DOCUMENT_3                                       1.067111
loan_income_ratio                                     1.181577
DAYS_LAST_PHONE_CHANGE                                1.198697
NAME_INCOME_TYPE_Working                              1.204270
AMT_CREDIT                                            1.232359
NAME_EDUCATION_TYPE_Higher education                  1.251945
REGION_RATING_CLIENT_W_CITY                           1.286967
DEF_30_CNT_SOCIAL_CIRCLE                              1.453538
days_employed_loan_ratio                              1.612204
FLAG_OWN_CAR_N                                        1.699333
DAYS_ID_PUBLISH                                       2.394581
DAYS_EMPLOYED                                         2.871785
CODE_GENDER_M                                         2.951471
AMT_ANNUITY                                           3.332789
AMT_GOODS_PRICE                                       4.781533
EXT_SOURCE_1                                          6.209192
DAYS_BIRTH                                            6.441286
EXT_SOURCE_2                                         12.152243
loan_annutiy_ratio                                   14.008465
EXT_SOURCE_3                                         17.136626
"""

############
# add loan_goods_price_ratio as feature
############
application_train['loan_goods_price_ratio']=application_train['AMT_CREDIT']/application_train['AMT_GOODS_PRICE']
application_test['loan_goods_price_ratio']=application_test['AMT_CREDIT']/application_test['AMT_GOODS_PRICE']

important_columns.append('loan_goods_price_ratio')

X = application_train[important_columns].fillna(0)
y = y_true

clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_2"+","+"4"+","+"GBC trained only on important columns of application data and feature added loan_goods_price_ratio"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)


"""
DAYS_LAST_PHONE_CHANGE                                1.098174
FLAG_DOCUMENT_3                                       1.110915
AMT_CREDIT                                            1.118040
NAME_INCOME_TYPE_Working                              1.181538
loan_income_ratio                                     1.263225
REGION_RATING_CLIENT_W_CITY                           1.316431
FLAG_WORK_PHONE                                       1.322289
days_employed_loan_ratio                              1.380329
NAME_EDUCATION_TYPE_Higher education                  1.423711
DEF_30_CNT_SOCIAL_CIRCLE                              1.474946
FLAG_OWN_CAR_N                                        1.565505
AMT_GOODS_PRICE                                       2.491946
DAYS_ID_PUBLISH                                       2.493969
DAYS_EMPLOYED                                         2.811480
CODE_GENDER_M                                         2.823271
AMT_ANNUITY                                           3.640253
loan_goods_price_ratio                                3.743451
EXT_SOURCE_1                                          5.773491
DAYS_BIRTH                                            5.962488
EXT_SOURCE_2                                         11.901724
loan_annutiy_ratio                                   14.425753
EXT_SOURCE_3                                         16.875368
"""

####
# what if days employed +ve is error
# looks like all +ve days employed are values 365243
####


