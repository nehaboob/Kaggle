#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:05:31 2018

@author: neha
refactor the code

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
import gc
import csv

#################################################
# load train and dev set skids
#################################################
def get_train_dev_sets():
    with open("train_all.txt", "rb") as fp:
        train_all_skid = pickle.load(fp)
    
    with open("dev_all.txt", "rb") as fp:
        dev_all_skid = pickle.load(fp)
        
    with open("dev_eyeball.txt", "rb") as fp:
        dev_eyeball_skid = pickle.load(fp)
        
    return train_all_skid, dev_all_skid, dev_eyeball_skid

###############################################
# from previous experiments, important columns
###############################################
def get_base_important_columns():
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
    
    return important_columns

###################################
# get False positive rate
###################################
def get_FPR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm[0][1]
    TN = cm[0][0]
    FPR = FP/(FP+TN)
    return FPR


def get_scores(y_true, y_predict, y_predict_proba, mode):
    scores_df = pd.Series()

    scores_df[mode+'_roc_auc'] = roc_auc_score(y_true, y_predict_proba)
    scores_df[mode+'_accuracy'] = accuracy_score(y_true, y_predict)
    scores_df[mode+'_recall'] = recall_score(y_true, y_predict)
    scores_df[mode+'_fpr'] = get_FPR(y_true, y_predict)
    scores_df[mode+'_precision'] = precision_score(y_true, y_predict)
    scores_df[mode+'_f1'] = f1_score(y_true, y_predict)
    
    return scores_df
####################################################################
# One-hot encoding for categorical columns with get_dummies
####################################################################
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = list(df.select_dtypes(['object']).columns)
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

#####################################################################
# get application train test features
####################################################################

def get_application_train_test():
    application_train = pd.read_csv('application_train.csv') # (307511, 122)
    application_test = pd.read_csv('application_test.csv') # (48744, 121)
        
    y_true = application_train[['SK_ID_CURR', 'TARGET']]
    del application_train['TARGET']
    
    mean_EXT_SOURCE_1 = application_train[~application_train.isnull()].EXT_SOURCE_1.mean()
    mean_EXT_SOURCE_3 = application_train[~application_train.isnull()].EXT_SOURCE_3.mean()
    
    application_train['loan_annutiy_ratio']=application_train['AMT_CREDIT']/application_train['AMT_ANNUITY']
    application_train['EXT_SOURCE_1'] = application_train['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
    application_train['EXT_SOURCE_3'] = application_train['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)
    
    application_test['loan_annutiy_ratio']=application_test['AMT_CREDIT']/application_test['AMT_ANNUITY']
    application_test['EXT_SOURCE_1'] = application_test['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
    application_test['EXT_SOURCE_3'] = application_test['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)
    
    x_object_columns = list(application_train.select_dtypes(['object']).columns)

    one_hot_df = pd.concat([application_train,application_test])
    one_hot_df = pd.get_dummies(one_hot_df, columns=x_object_columns)
    
    application_train = one_hot_df.iloc[:application_train.shape[0],:]
    application_test = one_hot_df.iloc[application_train.shape[0]:,]
    
    return application_train, application_test, y_true


#########################################
# bureau_features
#########################################
def get_bureau_features():
    
    bureau = pd.read_csv('bureau.csv')
    
    agg_dict= {'SK_ID_CURR': ['count'],
               'DAYS_CREDIT': ['max'],
               'CREDIT_DAY_OVERDUE': ['max'],
               'DAYS_CREDIT_ENDDATE': ['sum', 'min', 'max'],
               'DAYS_ENDDATE_FACT': ['sum', 'min', 'max'],
               'AMT_CREDIT_MAX_OVERDUE': ['sum'], 
               'CNT_CREDIT_PROLONG': ['sum'], 
               'AMT_CREDIT_SUM': ['sum'], 
               'AMT_CREDIT_SUM_DEBT': ['sum'], 
               'AMT_CREDIT_SUM_LIMIT': ['sum'], 
               'AMT_CREDIT_SUM_OVERDUE': ['sum'], 
               'AMT_ANNUITY': ['sum']
               }
    
    bureau_features = bureau.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
    bureau_features.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in bureau_features.columns]

    bureau_loan_status_total = bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE']).size().reset_index(name='total_loans')
    bureau_loan_status_total = bureau_loan_status_total.pivot(index='SK_ID_CURR', columns='CREDIT_ACTIVE', values='total_loans').fillna(0).reset_index()
    
    bureau_features = pd.merge(bureau_features, bureau_loan_status_total, on='SK_ID_CURR')

    del bureau, bureau_loan_status_total
    gc.collect()
    
    return bureau_features

def add_bureau_features(df, important_columns, mode):
    bureau_features = get_bureau_features()
    df = pd.merge(df, bureau_features, on='SK_ID_CURR', how='left')
    
    if mode != 'test':
        added_columns = list(bureau_features.columns)
        added_columns.remove('SK_ID_CURR')
        important_columns.extend(added_columns)
    
    return df, important_columns

#########################################
# bureau_balacne_features
#########################################
def get_bureau_balance_features():
    bureau_balance = pd.read_csv('bureau_balance.csv')
    
    agg_dict = {'SK_ID_BUREAU': ['count']}
    bureau_balance_agg = bureau_balance.groupby(['SK_ID_BUREAU', 'STATUS']).agg(agg_dict).reset_index()
    bureau_balance_agg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in bureau_balance_agg.columns]
    bureau_balance_agg = bureau_balance_agg.pivot(index='SK_ID_BUREAU', columns='STATUS', values='SK_ID_BUREAU_count').fillna(0).reset_index()
    
    
    del bureau_balance
    gc.collect()
    
    bureau = pd.read_csv('bureau.csv')
    
    bureau_balance_agg = pd.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], bureau_balance_agg, on='SK_ID_BUREAU')
    
    del bureau
    gc.collect()
    
    agg_dict = {'0': ['sum'], 
                '1': ['sum'],  
                '2': ['sum'], 
                '3': ['sum'],  
                '4': ['sum'], 
                '5': ['sum'], 
                'C': ['sum'],  
                'X': ['sum']  
                }
    bureau_balance_agg = bureau_balance_agg.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
    bureau_balance_agg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in bureau_balance_agg.columns]
    
    return bureau_balance_agg

def add_bureau_balance_features(df, important_columns, mode):
    bureau_balance_features = get_bureau_balance_features()
    df = pd.merge(df, bureau_balance_features, on='SK_ID_CURR', how='left')
    
    if mode != 'test':
        added_columns = list(bureau_balance_features.columns)
        added_columns.remove('SK_ID_CURR')
        important_columns.extend(added_columns)
    
    return df, important_columns

#########################################
# previous application
#########################################
# Preprocess previous_applications.csv
def get_previous_applications_features():
    prev = pd.read_csv('previous_application.csv')
    
    # Days 365.243 values -> nan
    """
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    """
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    prev_agg = prev.groupby('SK_ID_CURR').agg(num_aggregations)
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    prev_agg  = prev_agg.reset_index()
    
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    approved_agg  = approved_agg.reset_index()
    
    prev_agg = pd.merge(prev_agg, approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    refused_agg = refused_agg.reset_index()
    
    prev_agg = pd.merge(prev_agg, refused_agg, how='left', on='SK_ID_CURR')
    
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    
    return prev_agg

def add_previous_applications_features(df, important_columns, mode):
    previous_applications_features = get_previous_applications_features()
    df = pd.merge(df, previous_applications_features, on='SK_ID_CURR', how='left')
    
    if mode != 'test':
        added_columns = list(previous_applications_features.columns)
        added_columns.remove('SK_ID_CURR')
        important_columns.extend(added_columns)
    
    return df, important_columns


#########################################
# POS Cash
#########################################
def get_pos_cash_features():
    pos = pd.read_csv('POS_CASH_balance.csv')

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg = pos_agg.reset_index()

    del pos
    gc.collect()
    
    return pos_agg


def add_pos_cash_features(df, important_columns, mode):
    pos_cash_features = get_pos_cash_features()
    df = pd.merge(df, pos_cash_features, on='SK_ID_CURR', how='left')
    
    if mode != 'test':
        added_columns = list(pos_cash_features.columns)
        added_columns.remove('SK_ID_CURR')
        important_columns.extend(added_columns)
    
    return df, important_columns

######################################################
# Preprocess credit_card_balance.csv
######################################################
def get_credit_card_balance_features():
    cc = pd.read_csv('credit_card_balance.csv')

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    cc_agg = cc_agg.reset_index() 
 

    
    
    del cc
    gc.collect()
    
    return cc_agg

def add_credit_card_balance_features(df, important_columns, mode):
    credit_card_balance = get_credit_card_balance_features()
    df = pd.merge(df, credit_card_balance, on='SK_ID_CURR', how='left')
    
    if mode != 'test':
        added_columns = list(credit_card_balance.columns)
        added_columns.remove('SK_ID_CURR')
        important_columns.extend(added_columns)
    
    return df, important_columns

#########################################################
# Preprocess installments_payments.csv
########################################################
def get_installments_payments_features():
    ins = pd.read_csv('installments_payments.csv')

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = 0
    ins.loc[ins['AMT_INSTALMENT'] !=0, 'PAYMENT_PERC'] = ins.loc[ins['AMT_INSTALMENT'] !=0, 'AMT_PAYMENT'] / ins.loc[ins['AMT_INSTALMENT'] !=0, 'AMT_INSTALMENT']
   
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    ins_agg = ins_agg.reset_index()
    
    del ins
    gc.collect()
    
    return ins_agg

def add_installments_payments_features(df, important_columns, mode):
    installments_payments_features = get_installments_payments_features()
    df = pd.merge(df, installments_payments_features, on='SK_ID_CURR', how='left')
    
    if mode != 'test':
        added_columns = list(installments_payments_features.columns)
        added_columns.remove('SK_ID_CURR')
        important_columns.extend(added_columns)
    
    return df, important_columns


#########################################
# split train dev sets
#########################################
def get_train_dev_data(application_train, y_true, important_columns):
    
    train_all_skid, dev_all_skid, dev_eyeball_skid = get_train_dev_sets()
    X_train = application_train.loc[application_train.SK_ID_CURR.isin(train_all_skid), important_columns].fillna(0)
    y_train = y_true.loc[y_true.SK_ID_CURR.isin(train_all_skid), 'TARGET']
    X_dev = application_train.loc[application_train.SK_ID_CURR.isin(dev_all_skid), important_columns].fillna(0)
    y_dev = y_true.loc[y_true.SK_ID_CURR.isin(dev_all_skid), 'TARGET']

    return X_train, y_train,X_dev, y_dev

#########################################
# split train dev sets and train 
#########################################
def train_model(application_train, y_true, important_columns, model, version, estimators, desc):
    X_train, y_train, X_dev, y_dev = get_train_dev_data(application_train, y_true, important_columns)
    
    del application_train, y_true
    gc.collect()
    
    clf = GradientBoostingClassifier(n_estimators=estimators, random_state=0)
    clf.fit(X_train, y_train)

    ###################################################
    # results on the train set
    ###################################################
   
    y_train_predict = clf.predict(X_train)
    y_train_predict_proba = clf.predict_proba(X_train)
    
    scores_df = get_scores(y_train, y_train_predict, y_train_predict_proba[:, 1], 'train')
             
    ###################################################
    # results on the dev set
    ###################################################
    
    y_dev_predict = clf.predict(X_dev)
    y_dev_predict_proba = clf.predict_proba(X_dev)
    
    # get dev scores
    scores_df = scores_df.append(get_scores(y_dev, y_dev_predict, y_dev_predict_proba[:, 1], 'test'))
    
    with open('all_reasults.csv', 'a') as f:
        #f.write(model+","+version+","+"GBC trained only on important columns of application data with dev set CV adding bureau features"+","+"GradientBoostingClassifier"+","+str(clf)+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

        spamwriter = csv.writer(f)
        spamwriter.writerow([model,
                             version,
                             desc,
                             "GradientBoostingClassifier",
                             "GradientBoostingClassifier(n_estimators="+str(estimators)+" random_state=0)",
                             str(round(scores_df['test_roc_auc'], 4)),str(round(scores_df['test_accuracy'], 4)),
                             str(round(scores_df['test_recall'], 4)),str(round(scores_df['test_fpr'], 4)),
                             str(round(scores_df['test_precision'], 4)),str(round(scores_df['test_f1'], 4)),
                             "NA","NA","dev set",
                             str(round(scores_df['train_roc_auc'], 4)),str(round(scores_df['train_accuracy'], 4)),
                             str(round(scores_df['train_recall'], 4)),str(round(scores_df['train_fpr'], 4)),
                             str(round(scores_df['train_precision'], 4)),str(round(scores_df['train_f1'], 4))])

    
    important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)
    return clf,important_features

############################################
# train on all and generate submission file
############################################
def generate_submission_file(clf, application_train, application_test, important_columns, name):
    X = application_train[important_columns].fillna(0)
    y = y_true['TARGET']
    clf.fit(X, y)
    
    X_test_predict = application_test[important_columns].fillna(0)
    y_test_predict = clf.predict_proba(X_test_predict)
    
    submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
    submission.to_csv(name, index=False)

                            
#########################################
# 0 base for this experiment
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '1', 100)

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

#########################################
# 1  add_bureau_balance_features
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')

clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '2', 100)

"""
DAYS_LAST_PHONE_CHANGE                                1.003426
DAYS_ENDDATE_FACT_sum                                 1.099636
AMT_CREDIT                                            1.140826
DAYS_CREDIT_ENDDATE_sum                               1.144674
FLAG_OWN_CAR_Y                                        1.193120
FLAG_DOCUMENT_3                                       1.287723
NAME_INCOME_TYPE_Working                              1.290231
AMT_CREDIT_SUM_DEBT_sum                               1.310570
DEF_30_CNT_SOCIAL_CIRCLE                              1.345553
REGION_RATING_CLIENT_W_CITY                           1.399810
AMT_CREDIT_MAX_OVERDUE_sum                            1.408889
NAME_EDUCATION_TYPE_Higher education                  1.456289
DAYS_ID_PUBLISH                                       1.490545
DAYS_CREDIT_max                                       2.146775
AMT_CREDIT_SUM_OVERDUE_sum                            2.420376
CODE_GENDER_M                                         2.631301
DAYS_EMPLOYED                                         2.817329
AMT_ANNUITY                                           2.835955
Active                                                2.846246
AMT_GOODS_PRICE                                       3.548487
DAYS_BIRTH                                            5.830221
EXT_SOURCE_1                                          7.934033
loan_annutiy_ratio                                   10.806082
EXT_SOURCE_3                                         11.429304
EXT_SOURCE_2                                         11.807779
"""

#########################################
# 2  add_bureau_balance_features with sum and mean both 
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')

clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '3', 100)

"""
AMT_CREDIT_SUM_sum                       1.015029
FLAG_OWN_CAR_Y                           1.034648
DAYS_CREDIT_ENDDATE_sum                  1.161223
AMT_CREDIT_SUM_LIMIT_sum                 1.164588
AMT_CREDIT_MAX_OVERDUE_sum               1.262137
FLAG_DOCUMENT_3                          1.287253
DEF_30_CNT_SOCIAL_CIRCLE                 1.290417
NAME_INCOME_TYPE_Working                 1.293356
AMT_CREDIT_SUM_DEBT_sum                  1.336079
DAYS_ID_PUBLISH                          1.417924
REGION_RATING_CLIENT_W_CITY              1.571248
NAME_EDUCATION_TYPE_Higher education     1.628712
AMT_CREDIT                               1.834655
DAYS_CREDIT_max                          2.217091
CODE_GENDER_M                            2.230745
AMT_CREDIT_SUM_OVERDUE_sum               2.574189
Active                                   2.802435
AMT_GOODS_PRICE                          2.816091
DAYS_EMPLOYED                            3.216309
AMT_ANNUITY                              3.314308
DAYS_BIRTH                               5.554775
EXT_SOURCE_1                             7.956598
loan_annutiy_ratio                       9.400918
EXT_SOURCE_3                            11.077008
EXT_SOURCE_2                            11.892864
"""


#########################################
# 4  add_bureau_balance_features with sum and mean both with dummy na as category
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')

clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '4', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features dummy na as category")


#########################################
# 5 add_bureau_balance_features with add_previous_applications_features
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_previous_applications_features(application_train, important_columns, 'train')



clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '5', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features and add_previous_applications_features")


"""
FLAG_DOCUMENT_3                                       1.119345
NAME_INCOME_TYPE_Working                              1.129550
AMT_CREDIT                                            1.130403
DEF_30_CNT_SOCIAL_CIRCLE                              1.140261
NAME_FAMILY_STATUS_Married                            1.149268
DAYS_CREDIT_ENDDATE_sum                               1.205431
APPROVED_CNT_PAYMENT_MEAN                             1.219906
AMT_CREDIT_SUM_sum                                    1.229646
PREV_CNT_PAYMENT_MEAN                                 1.243619
DAYS_ID_PUBLISH                                       1.349181
NAME_EDUCATION_TYPE_Higher education                  1.459161
AMT_CREDIT_MAX_OVERDUE_sum                            1.586562
DAYS_CREDIT_max                                       2.077956
Active                                                2.115372
DAYS_EMPLOYED                                         2.231055
APPROVED_AMT_DOWN_PAYMENT_MAX                         2.237266
AMT_CREDIT_SUM_OVERDUE_sum                            2.292314
AMT_ANNUITY                                           2.459670
CODE_GENDER_M                                         2.602106
AMT_GOODS_PRICE                                       3.633107
DAYS_BIRTH                                            5.123254
EXT_SOURCE_1                                          6.973050
loan_annutiy_ratio                                    8.044944
EXT_SOURCE_3                                         10.145919
EXT_SOURCE_2                                         11.382782
"""

#########################################
# 6 add_bureau_balance_features with add_pos_cash_features
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_previous_applications_features(application_train, important_columns, 'train')
application_train, important_columns = add_pos_cash_features(application_train, important_columns, 'train')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '6', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features and add_previous_applications_features")


important_features.sort_values()


"""
REFUSED_CNT_PAYMENT_SUM                      1.030230
FLAG_OWN_CAR_N                               1.057204
DAYS_ID_PUBLISH                              1.088750
DEF_30_CNT_SOCIAL_CIRCLE                     1.111770
REGION_RATING_CLIENT_W_CITY                  1.240401
FLAG_DOCUMENT_3                              1.260123
APPROVED_CNT_PAYMENT_MEAN                    1.301496
PREV_CNT_PAYMENT_MEAN                        1.344771
AMT_CREDIT_SUM_sum                           1.440388
NAME_EDUCATION_TYPE_Higher education         1.524377
AMT_CREDIT                                   1.528783
DAYS_CREDIT_max                              1.842027
Active                                       1.948708
APPROVED_AMT_DOWN_PAYMENT_MAX                1.995362
DAYS_EMPLOYED                                2.170232
AMT_CREDIT_SUM_OVERDUE_sum                   2.207519
POS_MONTHS_BALANCE_SIZE                      2.211979
AMT_ANNUITY                                  2.312266
CODE_GENDER_M                                2.497027
AMT_GOODS_PRICE                              2.552275
POS_SK_DPD_DEF_MEAN                          3.218299
DAYS_BIRTH                                   4.946854
EXT_SOURCE_1                                 6.696697
loan_annutiy_ratio                           7.449252
EXT_SOURCE_3                                 9.672712
EXT_SOURCE_2                                10.052871
"""

#########################################
# main function for submission 
#########################################
application_train, application_test, y_true = get_application_train_test()
important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_previous_applications_features(application_train, important_columns, 'train')
application_train, important_columns = add_pos_cash_features(application_train, important_columns, 'train')

clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '7', 500, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features and add_previous_applications_features and POS features")

important_features.sort_values()

"""
REFUSED_CNT_PAYMENT_SUM                      1.030230
FLAG_OWN_CAR_N                               1.057204
DAYS_ID_PUBLISH                              1.088750
DEF_30_CNT_SOCIAL_CIRCLE                     1.111770
REGION_RATING_CLIENT_W_CITY                  1.240401
FLAG_DOCUMENT_3                              1.260123
APPROVED_CNT_PAYMENT_MEAN                    1.301496
PREV_CNT_PAYMENT_MEAN                        1.344771
AMT_CREDIT_SUM_sum                           1.440388
NAME_EDUCATION_TYPE_Higher education         1.524377
AMT_CREDIT                                   1.528783
DAYS_CREDIT_max                              1.842027
Active                                       1.948708
APPROVED_AMT_DOWN_PAYMENT_MAX                1.995362
DAYS_EMPLOYED                                2.170232
AMT_CREDIT_SUM_OVERDUE_sum                   2.207519
POS_MONTHS_BALANCE_SIZE                      2.211979
AMT_ANNUITY                                  2.312266
CODE_GENDER_M                                2.497027
AMT_GOODS_PRICE                              2.552275
POS_SK_DPD_DEF_MEAN                          3.218299
DAYS_BIRTH                                   4.946854
EXT_SOURCE_1                                 6.696697
loan_annutiy_ratio                           7.449252
EXT_SOURCE_3                                 9.672712
EXT_SOURCE_2                                10.052871
"""

#############################################
# generate submission
#############################################
application_test, important_columns = add_bureau_features(application_test, important_columns, 'test')
application_test, important_columns = add_bureau_balance_features(application_test, important_columns, 'test')
application_test, important_columns = add_previous_applications_features(application_test, important_columns, 'test')
application_test, important_columns = add_pos_cash_features(application_test, important_columns, 'test')

generate_submission_file(clf, application_train, application_test, important_columns, 'submission_11.csv')


#########################################
# 8 add_bureau_balance_features with add cc features
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_previous_applications_features(application_train, important_columns, 'train')
application_train, important_columns = add_pos_cash_features(application_train, important_columns, 'train')
application_train, important_columns = add_credit_card_balance_features(application_train, important_columns, 'train')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '8', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features and add_previous_applications_features and POS features and CC features")


important_features.sort_values()



#########################################
# 9 add_installments_payments_features with add cc features
#########################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_previous_applications_features(application_train, important_columns, 'train')
application_train, important_columns = add_pos_cash_features(application_train, important_columns, 'train')
application_train, important_columns = add_credit_card_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_installments_payments_features(application_train, important_columns, 'train')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '9', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features and add_previous_applications_features and POS features and CC features and prev installment features")


important_features.sort_values()

"""
NAME_INCOME_TYPE_Working                1.014744
POS_MONTHS_BALANCE_SIZE                 1.016770
INSTAL_PAYMENT_DIFF_MEAN                1.086065
CC_CNT_DRAWINGS_CURRENT_VAR             1.182542
NAME_EDUCATION_TYPE_Higher education    1.219057
REGION_RATING_CLIENT_W_CITY             1.428395
DAYS_CREDIT_max                         1.497366
APPROVED_AMT_DOWN_PAYMENT_MAX           1.572637
Active                                  1.589191
POS_SK_DPD_DEF_MEAN                     1.666896
PREV_CNT_PAYMENT_MEAN                   1.692000
CC_CNT_DRAWINGS_ATM_CURRENT_MEAN        1.797557
APPROVED_CNT_PAYMENT_MEAN               1.907532
AMT_CREDIT_SUM_OVERDUE_sum              1.942758
CODE_GENDER_M                           2.081049
DAYS_EMPLOYED                           2.133805
AMT_ANNUITY                             2.335699
INSTAL_AMT_PAYMENT_SUM                  2.740770
AMT_GOODS_PRICE                         3.033295
INSTAL_DPD_MEAN                         3.407610
DAYS_BIRTH                              4.030650
EXT_SOURCE_1                            5.578768
loan_annutiy_ratio                      6.475766
EXT_SOURCE_3                            8.002971
EXT_SOURCE_2                            9.429901
"""

#########################################
# 10 add_installments_payments_features with add cc features
#########################################
application_train, application_test, y_true = get_application_train_test()
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_previous_applications_features(application_train, important_columns, 'train')
application_train, important_columns = add_pos_cash_features(application_train, important_columns, 'train')
application_train, important_columns = add_credit_card_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_installments_payments_features(application_train, important_columns, 'train')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '10', 500, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features and add_previous_applications_features and POS features and CC features and prev installment features")


important_features.sort_values()

"""
NSTAL_PAYMENT_DIFF_MEAN              0.990310
DAYS_EMPLOYED                         1.059160
INSTAL_DAYS_ENTRY_PAYMENT_MAX         1.073045
DAYS_CREDIT_max                       1.079687
AMT_CREDIT_SUM_OVERDUE_sum            1.114748
AMT_GOODS_PRICE                       1.115239
INSTAL_DAYS_ENTRY_PAYMENT_SUM         1.187727
INSTAL_DPD_MEAN                       1.279210
Active                                1.329786
DAYS_CREDIT_ENDDATE_sum               1.368330
AMT_CREDIT_MAX_OVERDUE_sum            1.375106
AMT_CREDIT_SUM_sum                    1.471395
DAYS_CREDIT_ENDDATE_max               1.485496
INSTAL_AMT_PAYMENT_SUM                1.488114
APPROVED_CNT_PAYMENT_MEAN             1.522209
AMT_ANNUITY                           1.540504
DAYS_BIRTH                            2.822109
loan_annutiy_ratio                    4.440471
EXT_SOURCE_3                          4.505161
EXT_SOURCE_1                          4.524578
EXT_SOURCE_2                          4.982630
"""

#############################################
# generate submission
#############################################
application_test, important_columns = add_bureau_features(application_test, important_columns, 'test')
application_test, important_columns = add_bureau_balance_features(application_test, important_columns, 'test')
application_test, important_columns = add_previous_applications_features(application_test, important_columns, 'test')
application_test, important_columns = add_pos_cash_features(application_test, important_columns, 'test')
application_test, important_columns = add_credit_card_balance_features(application_test, important_columns, 'test')
application_test, important_columns = add_installments_payments_features(application_test, important_columns, 'test')


generate_submission_file(clf, application_train, application_test, important_columns, 'submission_12.csv')


################################################################################################
# 11 add all the missing features like one hot encoded and division features from the kernel
################################################################################################
application_train, application_test, y_true = get_application_train_test()
del application_test
gc.collect()

important_columns = get_base_important_columns()

application_train, important_columns = add_bureau_features(application_train, important_columns, 'train')
application_train, important_columns = add_bureau_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_previous_applications_features(application_train, important_columns, 'train')
application_train, important_columns = add_pos_cash_features(application_train, important_columns, 'train')
application_train, important_columns = add_credit_card_balance_features(application_train, important_columns, 'train')
application_train, important_columns = add_installments_payments_features(application_train, important_columns, 'train')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_10', '9', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau and bureau balance features and add_previous_applications_features and POS features and CC features and prev installment features")


important_features.sort_values()

