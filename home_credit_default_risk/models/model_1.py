#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:44:59 2018

@author: neha
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

"""
bureau = pd.read_csv('bureau.csv') # (1716428, 17)
bureau_balance = pd.read_csv('bureau_balance.csv') # (27299925, 3)

previous_application =  pd.read_csv('previous_application.csv') # (1670214, 37)
POS_CASH_balance =  pd.read_csv('POS_CASH_balance.csv') # (10001358, 8)
credit_card_balance = pd.read_csv('credit_card_balance.csv') # (3840312, 23) 
installments_payments = pd.read_csv('installments_payments.csv') #(13605401, 8)
"""

# simple first model using application_train
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

##################################################################################
# train only for application_train, remove the categorical columns
##################################################################################

# remove object columns from the training data
x_object_columns = list(application_train.select_dtypes(['object']).columns)
x_object_columns.extend(['SK_ID_CURR', 'TARGET'])
x_columns = list(application_train.columns.drop(x_object_columns))

X = application_train[x_columns].fillna(0)
y = y_true

# Gradient boosting  
clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=10, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

X_predict = application_test[x_columns].fillna(0)
y_predict = clf.predict_proba(X_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_predict[:, 1]})
submission.to_csv('submission_1.csv', index=False)

with open('all_reasults.csv', 'w+') as f:
    f.write("filename, version, description, algorithm, param, auc_roc, accuracy, tpr/recall, fpr, precision, f1_score \n")
    

with open('all_reasults.csv', 'a') as f:
    f.write("model_1"+","+"0"+","+"GBC trained only on non-object columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"\n")


####################################################################################
# train only for application_train, convert categorical columns to dummy variables 
####################################################################################

# get dummy variables for categorical variables
x_object_columns = list(application_train.select_dtypes(['object']).columns)

one_hot_df = pd.concat([application_train,application_test])
one_hot_df = pd.get_dummies(one_hot_df, columns=x_object_columns)

application_train = one_hot_df.iloc[:application_train.shape[0],:]
application_test = one_hot_df.iloc[application_train.shape[0]:,]

# drop columns 
x_columns = list(application_train.columns.drop(['SK_ID_CURR']))
X = application_train[x_columns].fillna(0)
y = y_true

# Gradient boosting  
clf = GradientBoostingClassifier(random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=10, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

X_predict = application_test[x_columns].fillna(0)
y_predict = clf.predict_proba(X_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_predict[:, 1]})
submission.to_csv('submission_2.csv', index=False)

with open('all_reasults.csv', 'a') as f:
    f.write("model_1"+","+"1"+","+"GBC trained only on all columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"submission_2.csv"+","+"\n")


####################################################################################
# train only for application_train, convert categorical columns to dummy variables 
# remove features where importance equals to 0 to reduce fit time - gives same CV score
# scince both train precision and recall are low - so there is a high bias in the model
# we are trying to increase model capacity (n_estimators=500) so it learns the train data better
####################################################################################

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


X = application_train[important_columns].fillna(0)
y = y_true

# Gradient boosting  
clf = GradientBoostingClassifier(n_estimators=500, random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

with open('all_reasults.csv', 'a') as f:
    f.write("model_1"+","+"2"+","+"GBC trained only on important columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500 random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+"\n")


####################################################################################
# train only for application_train, convert categorical columns to dummy variables 
# remove features where importance equals to 0 to reduce fit time - gives same CV score
# scince both train precision and recall are low - so there is a high bias in the model
# we are trying to increase model capacity (n_estimators=500) so it learns the train data better - no effect
# try changing other params of GBC to check if they give better results
####################################################################################

# Gradient boosting  
clf = GradientBoostingClassifier(n_estimators=100, random_state=0, max_features=9)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_1"+","+"3"+","+"GBC trained only on important columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=100 random_state=0 max_features=9)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+"\n")


# Gradient boosting  
clf = GradientBoostingClassifier(n_estimators=100, random_state=0, max_depth=5)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_1"+","+"4"+","+"GBC trained only on important columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=100 random_state=0 max_depth=5)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+"\n")

# Gradient boosting - submit resutls 
clf = GradientBoostingClassifier(n_estimators=500, random_state=0, max_depth=5)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_1"+","+"5"+","+"GBC trained only on important columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500 random_state=0 max_depth=5)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+"\n")

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

X_predict = application_test[important_columns].fillna(0)
y_predict = clf.predict_proba(X_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_predict[:, 1]})
submission.to_csv('submission_3.csv', index=False)

# Gradient boosting - more depth
clf = GradientBoostingClassifier(n_estimators=100, random_state=0, max_depth=10)
scores = cross_validate(clf, X, y, scoring=scoring, cv=4, return_train_score=True)
scores_df = pd.DataFrame(scores)
scores_df = scores_df.mean()

with open('all_reasults.csv', 'a') as f:
    f.write("model_1"+","+"6"+","+"GBC trained only on important columns of application data"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=100 random_state=0 max_depth=10)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"4"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

clf.fit(X, y)
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

X_predict = application_test[important_columns].fillna(0)
y_predict = clf.predict_proba(X_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_predict[:, 1]})
submission.to_csv('submission_4.csv', index=False)


#####################
# summary
# it is a case of high bias so
# Try either upscaling the minority class
# try feature engineering
# increase in model size is not taking care of bias, its is leading to overfitting
#####################


