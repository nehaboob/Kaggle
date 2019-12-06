#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:19:01 2018

@author: neharun the base algo again and again to get representative distribution of dev and train


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
# baseline model for this script
#############################################
application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']


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
    f.write("model_5"+","+"0"+","+"GBC trained only on important columns of application data with dev set CV"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")


##############################
# add one feature and check
##############################

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
    f.write("model_5"+","+"1"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

########################################
# add more trees and submit
########################################
application_test['loan_annutiy_ratio']=application_test['AMT_CREDIT']/application_test['AMT_ANNUITY']

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
    f.write("model_5"+","+"2"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500 random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

#############################################
# refit the model and make prediction
#############################################
X = application_train[important_columns].fillna(0)
y = y_true['TARGET']
clf.fit(X, y)

X_test_predict = application_test[important_columns].fillna(0)
y_test_predict = clf.predict_proba(X_test_predict)

submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
submission.to_csv('submission_6.csv', index=False)
                        

clf.feature_importances_
important_features = pd.Series(data=clf.feature_importances_*100,index=X.columns)

"""
CODE_GENDER_M                             1.028564
AMT_REQ_CREDIT_BUREAU_QRT                 1.225021
DEF_30_CNT_SOCIAL_CIRCLE                  1.230030
OWN_CAR_AGE                               1.278344
AMT_REQ_CREDIT_BUREAU_YEAR                1.618432
HOUR_APPR_PROCESS_START                   1.682993
REGION_POPULATION_RELATIVE                2.023786
DAYS_LAST_PHONE_CHANGE                    2.094573
AMT_ANNUITY                               2.695711
AMT_CREDIT                                2.855955
AMT_INCOME_TOTAL                          2.880275
DAYS_REGISTRATION                         3.123568
DAYS_EMPLOYED                             3.313121
DAYS_ID_PUBLISH                           3.324469
AMT_GOODS_PRICE                           3.347540
EXT_SOURCE_1                              5.191006
EXT_SOURCE_3                              6.811285
DAYS_BIRTH                                7.927198
EXT_SOURCE_2                             12.387758
loan_annutiy_ratio                       14.525530
"""
 
#####################################################
# add even more trees
#####################################################
clf = GradientBoostingClassifier(n_estimators=2000, random_state=0)
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
    f.write("model_5"+","+"3"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=2000 random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

########################################
# plot the roc curve 
########################################
fpr, tpr, thresholds = roc_curve(y_dev, y_dev_predict_proba[:, 1])
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % scores_df['test_roc_auc'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid(True)
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='--', color='r')
ax2.set_ylabel('thresholds', color='r')
ax2.set_ylim([thresholds[-1], thresholds[0]])
ax2.set_xlim([fpr[0], fpr[-1]])
plt.show()


#########################################################
# Try SMOTE
#########################################################
smot = SMOTE(random_state = 42)
X_train_smt, y_train_smt = smot.fit_sample(X_train, y_train)

clf = GradientBoostingClassifier(n_estimators=500, random_state=0)
clf.fit(X_train_smt, y_train_smt)

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
    f.write("model_5"+","+"4"+","+"GBC trained only on important columns of application data with dev set CV and loan_annutiy_ratio feature with SMOTE"+","+"GradientBoostingClassifier"+","+"GradientBoostingClassifier(n_estimators=500 random_state=0)"+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")


########################################
# plot the roc curve 
########################################
fpr, tpr, thresholds = roc_curve(y_dev, y_dev_predict_proba[:, 1])
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % scores_df['test_roc_auc'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid(True)
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='--', color='r')
ax2.set_ylabel('thresholds', color='r')
ax2.set_ylim([thresholds[-1], thresholds[0]])
ax2.set_xlim([fpr[0], fpr[-1]])
plt.show()
