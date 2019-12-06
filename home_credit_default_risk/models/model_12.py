#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:18:07 2018

@author: neha
"""

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
from sklearn.model_selection import KFold, StratifiedKFold

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

def get_application_train_test(nan_as_category=False):
    application_train = pd.read_csv('application_train.csv') # (307511, 122)
    application_test = pd.read_csv('application_test.csv') # (48744, 121)
                
    one_hot_df = application_train.append(application_test).reset_index()

    del application_train, application_test
    gc.collect()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    one_hot_df = one_hot_df[one_hot_df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        one_hot_df[bin_feature], uniques = pd.factorize(one_hot_df[bin_feature])
    
    one_hot_df, cat_cols = one_hot_encoder(one_hot_df, nan_as_category)
    
    mean_EXT_SOURCE_1 = one_hot_df[~one_hot_df.isnull()].EXT_SOURCE_1.mean()
    mean_EXT_SOURCE_3 = one_hot_df[~one_hot_df.isnull()].EXT_SOURCE_3.mean()
    
    one_hot_df['loan_annutiy_ratio']=one_hot_df['AMT_CREDIT']/one_hot_df['AMT_ANNUITY']
    one_hot_df['EXT_SOURCE_1'] = one_hot_df['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
    one_hot_df['EXT_SOURCE_3'] = one_hot_df['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)
        
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    one_hot_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    # Some simple new features (percentages)
    one_hot_df['DAYS_EMPLOYED_PERC'] = one_hot_df['DAYS_EMPLOYED'] / one_hot_df['DAYS_BIRTH']
    one_hot_df['INCOME_CREDIT_PERC'] = one_hot_df['AMT_INCOME_TOTAL'] / one_hot_df['AMT_CREDIT']
    one_hot_df['INCOME_PER_PERSON'] = one_hot_df['AMT_INCOME_TOTAL'] / one_hot_df['CNT_FAM_MEMBERS']
    one_hot_df['ANNUITY_INCOME_PERC'] = one_hot_df['AMT_ANNUITY'] / one_hot_df['AMT_INCOME_TOTAL']
    one_hot_df['PAYMENT_RATE'] = one_hot_df['AMT_ANNUITY'] / one_hot_df['AMT_CREDIT']
    
    return one_hot_df

#########################################
# bureau_features
#########################################
def get_bureau_features(nan_as_category = False):
    
    bureau = pd.read_csv('bureau.csv')
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    cat_aggregations = {}
    for cat in bureau_cat: 
        cat_aggregations[cat] = ['mean']
    
    agg_dict= {'SK_ID_CURR': ['count'],
               'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
               'CREDIT_DAY_OVERDUE': ['max', 'mean'],
               'DAYS_CREDIT_ENDDATE': ['sum', 'min', 'max', 'mean'],
               'DAYS_CREDIT_UPDATE': ['mean'],
               'DAYS_ENDDATE_FACT': ['sum', 'min', 'max'],
               'AMT_CREDIT_MAX_OVERDUE': ['sum', 'mean'], 
               'CNT_CREDIT_PROLONG': ['sum'], 
               'AMT_CREDIT_SUM': ['sum', 'max', 'mean'], 
               'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'], 
               'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean'], 
               'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean'], 
               'AMT_ANNUITY': ['sum', 'max', 'mean']
               }
    
    
    bureau_features = bureau.groupby('SK_ID_CURR').agg({**agg_dict, **cat_aggregations})
    bureau_features.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_features.columns.tolist()])
    bureau_features = bureau_features.reset_index()
    
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(agg_dict)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    active_agg = active_agg.reset_index()
    
    bureau_features = pd.merge(bureau_features, active_agg, how='left', on='SK_ID_CURR')
    
    del active, active_agg
    gc.collect()
    
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(agg_dict)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    closed_agg = closed_agg.reset_index()
    
    bureau_features = pd.merge(bureau_features, closed_agg, how='left', on='SK_ID_CURR')
    
    
    del bureau, closed, closed_agg
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
def get_bureau_balance_features(nan_as_category=False):
    bureau_balance = pd.read_csv('bureau_balance.csv')
    bureau_balance, bb_cat = one_hot_encoder(bureau_balance, nan_as_category)

    agg_dict = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        agg_dict[col] = ['mean', 'count']
        
    bureau_balance_agg = bureau_balance.groupby(['SK_ID_BUREAU']).agg(agg_dict).reset_index()
    bureau_balance_agg.columns = pd.Index(['BURO_BB_' + e[0] + "_" + e[1].upper() for e in bureau_balance_agg.columns.tolist()])

    
    del bureau_balance
    gc.collect()
    
    bureau = pd.read_csv('bureau.csv')
    
    bureau_balance_agg = pd.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], bureau_balance_agg, left_on='SK_ID_BUREAU', right_on='BURO_BB_SK_ID_BUREAU_', how='left')
    
    del bureau
    gc.collect()
    
    agg_dict = { 'BURO_BB_STATUS_0_MEAN': ['mean', 'sum'],          
                 'BURO_BB_STATUS_0_COUNT': ['mean', 'sum'],        
                 'BURO_BB_STATUS_1_MEAN': ['mean', 'sum'],        
                 'BURO_BB_STATUS_1_COUNT': ['mean', 'sum'],       
                 'BURO_BB_STATUS_2_MEAN': ['mean', 'sum'],         
                 'BURO_BB_STATUS_2_COUNT': ['mean', 'sum'],   
                 'BURO_BB_STATUS_3_MEAN': ['mean', 'sum'],         
                 'BURO_BB_STATUS_3_COUNT': ['mean', 'sum'],       
                 'BURO_BB_STATUS_4_MEAN': ['mean', 'sum'],     
                 'BURO_BB_STATUS_4_COUNT': ['mean', 'sum'],      
                 'BURO_BB_STATUS_5_MEAN': ['mean', 'sum'],     
                 'BURO_BB_STATUS_5_COUNT': ['mean', 'sum'],     
                 'BURO_BB_STATUS_C_MEAN': ['mean', 'sum'],   
                 'BURO_BB_STATUS_C_COUNT': ['mean', 'sum'],    
                 'BURO_BB_STATUS_X_MEAN': ['mean', 'sum'],   
                 'BURO_BB_STATUS_X_COUNT': ['mean', 'sum'],   
                 'BURO_BB_MONTHS_BALANCE_MIN': ['min'],
                 'BURO_BB_MONTHS_BALANCE_MAX': ['max'],
                 'BURO_BB_MONTHS_BALANCE_SIZE': ['mean', 'sum']
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
def get_previous_applications_features(nan_as_category=False):
    prev = pd.read_csv('previous_application.csv')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)

    # Days 365.243 values -> nan
    
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    
    prev['APP_CREDIT_PERC'] = 0
    prev.loc[prev['AMT_CREDIT'] != 0, 'APP_CREDIT_PERC'] = prev.loc[prev['AMT_CREDIT'] != 0,'AMT_APPLICATION'] / prev.loc[prev['AMT_CREDIT'] !=0, 'AMT_CREDIT']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    prev_agg  = prev_agg.reset_index()

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    approved_agg  = approved_agg.reset_index()

    prev_agg = pd.merge(prev_agg, approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
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
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
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
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

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
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)

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

    for cat in cat_cols:
        aggregations[cat] = ['mean']
        
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

def GBC_train(X_train, y_train, X_dev, y_dev, estimators):
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
    
    return clf, scores_df

def LGBM_train(X_train, y_train, X_dev, y_dev, estimators):
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

    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_dev, y_dev)], eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

    ###################################################
    # results on the train set
    ###################################################
 
    y_train_predict_proba = clf.predict_proba(X_train, num_iteration=clf.best_iteration)
    y_train_predict = (y_train_predict_proba[:, 1] > 0.5)*1 
                      
    scores_df = get_scores(y_train, y_train_predict, y_train_predict_proba[:, 1], 'train')
                  
    ###################################################
    # results on the dev set
    ###################################################
    y_dev_predict_proba = clf.predict_proba(X_dev, num_iteration=clf.best_iteration)
    y_dev_predict = (y_dev_predict_proba[:, 1] > 0.5)*1 

    # get dev scores
    scores_df = scores_df.append(get_scores(y_dev, y_dev_predict, y_dev_predict_proba[:, 1], 'test'))
    
    return clf, scores_df
    

#########################################
# split train dev sets and train 
#########################################
def train_model(application_train, y_true, important_columns, model, version, estimators, desc, algo='GBC'):
    X_train, y_train, X_dev, y_dev = get_train_dev_data(application_train, y_true, important_columns)
    
    del application_train, y_true
    gc.collect()
    
    if algo == 'LGB':
        clf, scores_df = LGBM_train(X_train, y_train, X_dev, y_dev, estimators)
    else:
        clf, scores_df = GBC_train(X_train, y_train, X_dev, y_dev, estimators)

    with open('all_reasults.csv', 'a') as f:
        #f.write(model+","+version+","+"GBC trained only on important columns of application data with dev set CV adding bureau features"+","+"GradientBoostingClassifier"+","+str(clf)+","+str(round(scores_df['test_roc_auc'], 4))+","+str(round(scores_df['test_accuracy'], 4))+","+str(round(scores_df['test_recall'], 4))+","+str(round(scores_df['test_fpr'], 4))+","+str(round(scores_df['test_precision'], 4))+","+ str(round(scores_df['test_f1'], 4))+","+"NA"+","+"NA"+","+"dev set"+","+str(round(scores_df['train_roc_auc'], 4))+","+str(round(scores_df['train_accuracy'], 4))+","+str(round(scores_df['train_recall'], 4))+","+str(round(scores_df['train_fpr'], 4))+","+str(round(scores_df['train_precision'], 4))+","+ str(round(scores_df['train_f1'], 4))+"\n")

        spamwriter = csv.writer(f)
        spamwriter.writerow([model,
                             version,
                             desc,
                             algo,
                             algo+"(n_estimators="+str(estimators)+" random_state=0)",
                             str(round(scores_df['test_roc_auc'], 4)),str(round(scores_df['test_accuracy'], 4)),
                             str(round(scores_df['test_recall'], 4)),str(round(scores_df['test_fpr'], 4)),
                             str(round(scores_df['test_precision'], 4)),str(round(scores_df['test_f1'], 4)),
                             "NA","NA","dev set",
                             str(round(scores_df['train_roc_auc'], 4)),str(round(scores_df['train_accuracy'], 4)),
                             str(round(scores_df['train_recall'], 4)),str(round(scores_df['train_fpr'], 4)),
                             str(round(scores_df['train_precision'], 4)),str(round(scores_df['train_f1'], 4))])

    
    important_features = pd.Series(data=clf.feature_importances_*100,index=X_train.columns)
    return clf,important_features



###################################################
# K flod LGB
###################################################
# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

def kfold_lightgbm(df, num_folds, submission_file_name, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
        
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
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

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()


    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    oof_preds_pred = (oof_preds > 0.5)*1
    scores_df = get_scores(train_df['TARGET'], oof_preds_pred, oof_preds, 'test')
    print(scores_df)
    
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

    return feature_importance_df



############################################
# train on all and generate submission file
############################################
def generate_submission_file(clf, application_train, application_test, y_true, important_columns, name):
    X = application_train[important_columns].fillna(0)
    y = y_true['TARGET']
    clf.fit(X, y, eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
    
    X_test_predict = application_test[important_columns].fillna(0)
    y_test_predict = clf.predict_proba(X_test_predict, num_iteration=clf.best_iteration_)
    
    submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
    submission.to_csv(name, index=False)

                            
#########################################
# 0 base for this experiment - add all the missing features like one hot encoded and division features from the kernel
# only bureau features
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')

application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '0', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau features all")


important_features.sort_values()

"""
DEF_30_CNT_SOCIAL_CIRCLE                       1.023356
BURO_AMT_CREDIT_MAX_OVERDUE_MEAN               1.223059
BURO_DAYS_CREDIT_MEAN                          1.380067
BURO_CREDIT_TYPE_Microloan_MEAN                1.414192
REGION_RATING_CLIENT_W_CITY                    1.454203
BURO_AMT_CREDIT_SUM_DEBT_MEAN                  1.599116
ACTIVE_DAYS_CREDIT_MAX                         1.602728
NAME_EDUCATION_TYPE_Higher education           1.628132
AMT_CREDIT                                     1.788963
OWN_CAR_AGE                                    1.817727
ACTIVE_SK_ID_CURR_COUNT                        2.117912
AMT_ANNUITY                                    2.474352
DAYS_EMPLOYED                                  3.526176
AMT_GOODS_PRICE                                3.817006
DAYS_BIRTH                                     4.977226
EXT_SOURCE_1                                   8.388675
loan_annutiy_ratio                             9.286674
EXT_SOURCE_3                                  10.133834
EXT_SOURCE_2                                  11.177014
"""

#########################################
# 1 bureau features and BB features
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')

application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '1', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau features all and BB features all")


important_features.sort_values()

"""
OCCUPATION_TYPE_Laborers                                               1.010547
DAYS_ID_PUBLISH                                                        1.047759
FLAG_DOCUMENT_3                                                        1.047930
BURO_AMT_CREDIT_SUM_OVERDUE_MEAN                                       1.068397
NAME_INCOME_TYPE_Working                                               1.074312
OCCUPATION_TYPE_Drivers                                                1.078724
BURO_DAYS_CREDIT_MEAN                                                  1.095216
ACTIVE_DAYS_CREDIT_MAX                                                 1.123765
DEF_30_CNT_SOCIAL_CIRCLE                                               1.163345
BURO_CREDIT_TYPE_Microloan_MEAN                                        1.232984
BURO_AMT_CREDIT_MAX_OVERDUE_MEAN                                       1.304107
AMT_CREDIT                                                             1.365872
BURO_DAYS_CREDIT_MAX                                                   1.389063
REGION_RATING_CLIENT_W_CITY                                            1.600999
NAME_EDUCATION_TYPE_Higher education                                   1.650669
BURO_AMT_CREDIT_SUM_DEBT_MEAN                                          1.838358
OWN_CAR_AGE                                                            1.848730
ACTIVE_SK_ID_CURR_COUNT                                                2.212615
AMT_ANNUITY                                                            2.352024
DAYS_EMPLOYED                                                          3.022012
AMT_GOODS_PRICE                                                        3.514911
DAYS_BIRTH                                                             4.650059
EXT_SOURCE_1                                                           7.711711
loan_annutiy_ratio                                                     8.651298
EXT_SOURCE_3                                                          10.045459
EXT_SOURCE_2                                                          11.092001
"""


#########################################
# 2 bureau features and BB features and prev app
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')

application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '2', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau features all and BB features all and prev app all")


important_features.sort_values()

"""
BURO_DAYS_CREDIT_MEAN                                  1.121175
BURO_DAYS_CREDIT_MAX                                   1.150124
PREV_NAME_PRODUCT_TYPE_walk-in_MEAN                    1.157830
REGION_RATING_CLIENT_W_CITY                            1.168276
OWN_CAR_AGE                                            1.183286
APPROVED_AMT_ANNUITY_MAX                               1.188700
BURO_AMT_CREDIT_SUM_DEBT_MEAN                          1.200569
PREV_NAME_YIELD_GROUP_high_MEAN                        1.271308
FLAG_DOCUMENT_3                                        1.320562
NAME_EDUCATION_TYPE_Higher education                   1.446158
PREV_NAME_CONTRACT_STATUS_Refused_MEAN                 1.515604
APPROVED_AMT_DOWN_PAYMENT_MAX                          1.542552
ACTIVE_DAYS_CREDIT_MAX                                 1.606446
DAYS_EMPLOYED                                          2.093022
PREV_CNT_PAYMENT_MEAN                                  2.231705
AMT_ANNUITY                                            2.694950
AMT_GOODS_PRICE                                        3.454262
DAYS_BIRTH                                             3.473695
EXT_SOURCE_1                                           6.718583
loan_annutiy_ratio                                     6.858914
EXT_SOURCE_3                                           8.313295
EXT_SOURCE_2                                           9.911698
"""



#########################################
# 3 bureau features and BB features and prev app and pos cash
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')

application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '3', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau features all and BB features all and prev app all and pos cash all")


important_features.sort_values()

"""
PREV_NAME_YIELD_GROUP_high_MEAN                                          1.022836
BURO_DAYS_CREDIT_MAX                                                     1.047442
BURO_CREDIT_TYPE_Microloan_MEAN                                          1.074427
AMT_CREDIT                                                               1.077124
REGION_RATING_CLIENT_W_CITY                                              1.103748
NAME_INCOME_TYPE_Working                                                 1.152845
PREV_NAME_PRODUCT_TYPE_walk-in_MEAN                                      1.182577
ACTIVE_AMT_CREDIT_SUM_OVERDUE_SUM                                        1.238778
APPROVED_AMT_DOWN_PAYMENT_MAX                                            1.245381
NAME_EDUCATION_TYPE_Higher education                                     1.376420
PREV_NAME_CONTRACT_STATUS_Refused_MEAN                                   1.385067
BURO_DAYS_CREDIT_MEAN                                                    1.393037
POS_MONTHS_BALANCE_SIZE                                                  1.417049
DAYS_EMPLOYED                                                            2.315461
AMT_ANNUITY                                                              2.448997
PREV_CNT_PAYMENT_MEAN                                                    2.780113
POS_SK_DPD_DEF_MEAN                                                      3.063350
AMT_GOODS_PRICE                                                          3.189047
DAYS_BIRTH                                                               3.445445
loan_annutiy_ratio                                                       5.841986
EXT_SOURCE_1                                                             6.184671
EXT_SOURCE_3                                                             8.386232
EXT_SOURCE_2                                                             9.652085
"""

#########################################
# 4 bureau features and BB features and prev app and pos cash and cc
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')

application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '4', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau features all and BB features all and prev app all and pos cash all and cc all")


important_features.sort_values()

"""
BURO_DAYS_CREDIT_MEAN                                                    1.043978
ACTIVE_SK_ID_CURR_COUNT                                                  1.068576
PREV_NAME_YIELD_GROUP_high_MEAN                                          1.123950
APPROVED_AMT_DOWN_PAYMENT_MAX                                            1.139563
REGION_RATING_CLIENT_W_CITY                                              1.180875
CC_CNT_DRAWINGS_CURRENT_VAR                                              1.187689
APPROVED_CNT_PAYMENT_MEAN                                                1.261882
POS_MONTHS_BALANCE_SIZE                                                  1.347528
NAME_EDUCATION_TYPE_Higher education                                     1.393772
CC_CNT_DRAWINGS_ATM_CURRENT_MEAN                                         1.414943
ACTIVE_DAYS_CREDIT_MAX                                                   1.503121
PREV_NAME_CONTRACT_STATUS_Refused_MEAN                                   1.599516
PREV_CNT_PAYMENT_MEAN                                                    1.659450
DAYS_EMPLOYED                                                            2.316624
AMT_ANNUITY                                                              2.322084
POS_SK_DPD_DEF_MEAN                                                      2.756419
AMT_GOODS_PRICE                                                          3.064805
DAYS_BIRTH                                                               3.227587
EXT_SOURCE_1                                                             5.521055
loan_annutiy_ratio                                                       6.639125
EXT_SOURCE_3                                                             7.623228
EXT_SOURCE_2                                                             9.380476
"""

#########################################
# 5 bureau features and BB features and prev app and pos cash and cc and installment all
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_installments_payments_features(whole_df, important_columns, 'train')


application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '5', 100, 
                                      "GBC trained only on important columns of application data with dev set CV adding bureau features all and BB features all and prev app all and pos cash all and cc all and installment all")


important_features.sort_values()

"""
BURO_DAYS_CREDIT_MEAN                                     1.043075
NAME_EDUCATION_TYPE_Higher education                      1.194898
CC_CNT_DRAWINGS_CURRENT_VAR                               1.270426
REGION_RATING_CLIENT_W_CITY                               1.323047
ACTIVE_DAYS_CREDIT_MAX                                    1.443229
POS_SK_DPD_DEF_MEAN                                       1.567308
PREV_NAME_CONTRACT_STATUS_Refused_MEAN                    1.646717
CC_CNT_DRAWINGS_ATM_CURRENT_MEAN                          1.666682
PREV_CNT_PAYMENT_MEAN                                     2.107945
DAYS_EMPLOYED                                             2.123667
AMT_ANNUITY                                               2.208774
INSTAL_AMT_PAYMENT_SUM                                    2.409566
AMT_GOODS_PRICE                                           2.608515
INSTAL_DPD_MEAN                                           2.799070
DAYS_BIRTH                                                3.350315
EXT_SOURCE_1                                              5.809719
loan_annutiy_ratio                                        5.924581
EXT_SOURCE_3                                              7.608894
EXT_SOURCE_2                                              9.503091
"""


#########################################
# 6 train for all the features
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_installments_payments_features(whole_df, important_columns, 'train')


application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

important_columns = list(whole_df.columns)
important_columns.remove('SK_ID_CURR')

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '6', 100, 
                                      "GBC trained for ALL the features")


important_features.sort_values()

"""
BURO_AMT_CREDIT_SUM_DEBT_MEAN                             1.027856
ANNUITY_INCOME_PERC                                       1.030802
BURO_DAYS_CREDIT_MEAN                                     1.059896
CC_CNT_DRAWINGS_CURRENT_VAR                               1.068712
DAYS_EMPLOYED                                             1.096011
APPROVED_CNT_PAYMENT_MEAN                                 1.114283
REGION_RATING_CLIENT_W_CITY                               1.143253
NAME_EDUCATION_TYPE_Higher education                      1.160270
BURO_DAYS_CREDIT_MAX                                      1.226548
APPROVED_AMT_DOWN_PAYMENT_MAX                             1.242617
ACTIVE_SK_ID_CURR_COUNT                                   1.338647
CC_CNT_DRAWINGS_ATM_CURRENT_MEAN                          1.460410
AMT_ANNUITY                                               1.478778
POS_SK_DPD_DEF_MEAN                                       1.512387
PREV_NAME_CONTRACT_STATUS_Refused_MEAN                    1.645391
PREV_CNT_PAYMENT_MEAN                                     1.967755
AMT_GOODS_PRICE                                           2.015764
INSTAL_DPD_MEAN                                           2.265948
PAYMENT_RATE                                              2.314156
INSTAL_AMT_PAYMENT_SUM                                    2.384689
CODE_GENDER                                               2.413114
DAYS_BIRTH                                                3.513506
loan_annutiy_ratio                                        3.907150
EXT_SOURCE_1                                              5.622773
EXT_SOURCE_3                                              7.589903
EXT_SOURCE_2                                              9.050715
dtype: float64
"""

#########################################
# 7 train for all the features - 500 estimators
#########################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_installments_payments_features(whole_df, important_columns, 'train')


application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
del application_test
gc.collect()

important_columns = list(whole_df.columns)
important_columns.remove('SK_ID_CURR')
important_columns.remove('index')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '7', 500, 
                                      "GBC trained for ALL the features")


important_features.sort_values()

# save the model to disk
filename = 'GBC_500_all_features_model_12.sav'
pickle.dump(clf, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))

"""
BURO_CREDIT_TYPE_Microloan_MEAN                     0.629711
INSTAL_PAYMENT_DIFF_MEAN                            0.646232
ACTIVE_DAYS_CREDIT_ENDDATE_MIN                      0.658865
ACTIVE_DAYS_CREDIT_MAX                              0.674102
CODE_GENDER                                         0.701394
INSTAL_DAYS_ENTRY_PAYMENT_MAX                       0.703179
CC_CNT_DRAWINGS_CURRENT_VAR                         0.724761
AMT_GOODS_PRICE                                     0.727519
BURO_AMT_CREDIT_MAX_OVERDUE_MEAN                    0.731219
POS_SK_DPD_DEF_MEAN                                 0.731450
INSTAL_DPD_MEAN                                     0.740995
BURO_BB_STATUS_1_MEAN_mean                          0.742032
DAYS_ID_PUBLISH                                     0.783645
DAYS_EMPLOYED_PERC                                  0.806602
INSTAL_DAYS_ENTRY_PAYMENT_MEAN                      0.841358
ACTIVE_SK_ID_CURR_COUNT                             0.873608
INSTAL_AMT_PAYMENT_SUM                              0.976699
AMT_ANNUITY                                         0.985415
APPROVED_CNT_PAYMENT_MEAN                           1.025379
loan_annutiy_ratio                                  1.800899
PAYMENT_RATE                                        2.353359
DAYS_BIRTH                                          2.530452
EXT_SOURCE_1                                        3.226399
EXT_SOURCE_3                                        3.890863
EXT_SOURCE_2                                        4.205281
"""


#############################################
# 8 lightGBP for all the columns
#############################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_installments_payments_features(whole_df, important_columns, 'train')


application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'] 
gc.collect()

important_columns = list(application_train.columns)
important_columns.remove('SK_ID_CURR')
important_columns.remove('index')

clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '9', 10000, 
                                      "LGB trained for ALL the features", 'LGB')

important_features.sort_values()

X_test_predict = application_test[important_columns].fillna(0)

y_test_predict = clf.predict_proba(X_test_predict, num_iteration=clf.best_iteration)
    
submission = pd.DataFrame({'SK_ID_CURR': application_test['SK_ID_CURR'], 'TARGET': y_test_predict[:, 1]})
submission.to_csv('submission_13.csv', index=False)


#######################################################
# LightGBM with less features 
#######################################################
important_features_org = important_features

important_features_org[important_features_org > 0.05].sum() 
(important_features_org > 0.05).sum()

important_columns= list(important_features_org[important_features_org > 0.05].index)


clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '10', 10000, 
                                      "LGB trained for 412 the features", 'LGB')

important_features.sort_values()


#######################################################
# LightGBM with less features 
#######################################################
important_features_org = important_features

important_features_org[important_features_org > 0.1].sum() 
(important_features_org > 0.1).sum()

important_columns= list(important_features_org[important_features_org > 0.1].index)


clf, important_features = train_model(application_train, y_true, important_columns, 'model_12', '11', 10000, 
                                      "LGB trained for 412 the features", 'LGB')

important_features.sort_values()

#####################################################
# LGM with 5 folds
#####################################################
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_installments_payments_features(whole_df, important_columns, 'train')

feat_importance = kfold_lightgbm(whole_df, num_folds= 5, submission_file_name='submission_14.csv', stratified= False, debug=False)


