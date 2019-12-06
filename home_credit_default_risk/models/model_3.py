#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:46:29 2018

@author: neha

Try upsampling the minority class

"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, confusion_matrix
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler  


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
    f.write("model_3"+","+"0"+","+"GBC trained only on important columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

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


######################################
# over sampling using imblearn
######################################
X = application_train[important_columns].fillna(0)
y = y_true

ros = RandomOverSampler(ratio='auto', random_state=42);  
X_res, y_res = ros.fit_sample(X, y); 
   
clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X_res, y_res, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()
                             
with open('all_reasults.csv', 'a') as f:
    f.write("model_3"+","+"1"+","+"GBC trained only on important columns of application data with equal upsampling"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X_res, y_res)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)
                             
"""
NAME_INCOME_TYPE_Working                              1.089460
DAYS_REGISTRATION                                     1.214227
FLAG_DOCUMENT_3                                       1.406602
NAME_FAMILY_STATUS_Married                            1.552303
DEF_30_CNT_SOCIAL_CIRCLE                              1.594723
DAYS_LAST_PHONE_CHANGE                                1.660961
FLAG_OWN_CAR_N                                        1.716983
OWN_CAR_AGE                                           1.724698
REGION_RATING_CLIENT_W_CITY                           1.781353
AMT_REQ_CREDIT_BUREAU_YEAR                            1.785188
NAME_EDUCATION_TYPE_Higher education                  1.867088
CODE_GENDER_M                                         2.576588
DAYS_ID_PUBLISH                                       3.158141
DAYS_EMPLOYED                                         4.360459
AMT_ANNUITY                                           4.628441
AMT_CREDIT                                            5.531980
AMT_GOODS_PRICE                                       5.852874
EXT_SOURCE_1                                          5.913724
DAYS_BIRTH                                            6.797801
EXT_SOURCE_2                                         11.566745
EXT_SOURCE_3                                         16.504820
"""

X_predict = application_test[important_columns].fillna(0)
y_predict = clf.predict_proba(X_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_predict[:, 1]})
submission.to_csv('submission_5.csv', index=False)
                        

#######################################
# try different sampling ratio
#######################################
X = application_train[important_columns].fillna(0)
y = y_true

ros = RandomOverSampler(ratio={0:282686, 1:124125}, random_state=42);  
X_res, y_res = ros.fit_sample(X, y); 
   
clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X_res, y_res, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()
                             
with open('all_reasults.csv', 'a') as f:
    f.write("model_3"+","+"2"+","+"GBC trained only on important columns of application data with upsampling 1_class*4 ratio"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X_res, y_res)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)
 
"""
FLAG_OWN_CAR_Y                            1.101788
DAYS_REGISTRATION                         1.120281
NAME_FAMILY_STATUS_Married                1.202630
FLAG_OWN_CAR_N                            1.261891
FLAG_DOCUMENT_3                           1.338313
NAME_INCOME_TYPE_Working                  1.436056
OWN_CAR_AGE                               1.470174
DEF_30_CNT_SOCIAL_CIRCLE                  1.597571
DAYS_LAST_PHONE_CHANGE                    1.634256
REGION_RATING_CLIENT_W_CITY               1.820183
NAME_EDUCATION_TYPE_Higher education      1.845972
CODE_GENDER_M                             2.554243
DAYS_ID_PUBLISH                           2.821757
AMT_ANNUITY                               4.053180
DAYS_EMPLOYED                             4.295864
AMT_CREDIT                                4.674766
EXT_SOURCE_1                              7.061763
DAYS_BIRTH                                7.378910
AMT_GOODS_PRICE                           8.818930
EXT_SOURCE_2                             11.638719
EXT_SOURCE_3                             16.792044
"""

                             