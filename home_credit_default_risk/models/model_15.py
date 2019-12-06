#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 20:12:07 2018

@author: neha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import make_scorer, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import lightgbm as lgb
import xgboost as xgb
import random
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier, cv
import gc
import csv
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats
from sklearn.linear_model import LogisticRegression


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

##################################################################
# measure the slope of the time series
##################################################################
def linear_fit(df):
    col_x = df.columns[0]
    col_y = df.columns[1]
    
    y = df[col_x].values
    x = df[col_y].values
   
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    return slope
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
        
    mean_EXT_SOURCE_1 = one_hot_df[~one_hot_df.isnull()].EXT_SOURCE_1.mean()
    mean_EXT_SOURCE_3 = one_hot_df[~one_hot_df.isnull()].EXT_SOURCE_3.mean()
    
    one_hot_df['EXT_SOURCE_1'] = one_hot_df['EXT_SOURCE_1'].fillna(mean_EXT_SOURCE_1)
    one_hot_df['EXT_SOURCE_3'] = one_hot_df['EXT_SOURCE_3'].fillna(mean_EXT_SOURCE_3)
        
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    one_hot_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    """
    one_hot_df.loc[one_hot_df['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    one_hot_df.loc[one_hot_df['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    one_hot_df.loc[one_hot_df['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    one_hot_df.loc[one_hot_df['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    one_hot_df.loc[one_hot_df['OBS_30_CNT_SOCIAL_CIRCLE'] > 40, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    """
    
        
    docs = [_f for _f in one_hot_df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in one_hot_df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
   
    inc_by_org = one_hot_df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    # Some simple new features (percentages)
    #one_hot_df['loan_annutiy_ratio']=one_hot_df['AMT_CREDIT']/one_hot_df['AMT_ANNUITY']
    #one_hot_df['DAYS_EMPLOYED_PERC'] = one_hot_df['DAYS_EMPLOYED'] / one_hot_df['DAYS_BIRTH']
    #one_hot_df['INCOME_CREDIT_PERC'] = one_hot_df['AMT_INCOME_TOTAL'] / one_hot_df['AMT_CREDIT']
    one_hot_df['INCOME_PER_PERSON'] = one_hot_df['AMT_INCOME_TOTAL'] / one_hot_df['CNT_FAM_MEMBERS']
    #one_hot_df['ANNUITY_INCOME_PERC'] = one_hot_df['AMT_ANNUITY'] / one_hot_df['AMT_INCOME_TOTAL']
    one_hot_df['PAYMENT_RATE'] = one_hot_df['AMT_ANNUITY'] / one_hot_df['AMT_CREDIT']
    one_hot_df['CHILDREN_RATIO'] = one_hot_df['CNT_CHILDREN'] / one_hot_df['CNT_FAM_MEMBERS']
    
    one_hot_df['NEW_CREDIT_TO_GOODS_RATIO'] = one_hot_df['AMT_CREDIT'] / one_hot_df['AMT_GOODS_PRICE']
    one_hot_df['NEW_DOC_IND_KURT'] = one_hot_df[docs].kurtosis(axis=1)
    one_hot_df['NEW_LIVE_IND_SUM'] = one_hot_df[live].sum(axis=1)
    one_hot_df['NEW_INC_PER_CHLD'] = one_hot_df['AMT_INCOME_TOTAL'] / (1 + one_hot_df['CNT_CHILDREN'])
    one_hot_df['NEW_INC_BY_ORG'] = one_hot_df['ORGANIZATION_TYPE'].map(inc_by_org)
    one_hot_df['NEW_EMPLOY_TO_BIRTH_RATIO'] = one_hot_df['DAYS_EMPLOYED'] / one_hot_df['DAYS_BIRTH']
    one_hot_df['NEW_ANNUITY_TO_INCOME_RATIO'] = one_hot_df['AMT_ANNUITY'] / (1 + one_hot_df['AMT_INCOME_TOTAL'])
    one_hot_df['NEW_SOURCES_PROD'] = one_hot_df['EXT_SOURCE_1'] * one_hot_df['EXT_SOURCE_2'] * one_hot_df['EXT_SOURCE_3']
    one_hot_df['NEW_EXT_SOURCES_MEAN'] = one_hot_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    one_hot_df['NEW_SCORES_STD'] = one_hot_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    one_hot_df['NEW_SCORES_STD'] = one_hot_df['NEW_SCORES_STD'].fillna(one_hot_df['NEW_SCORES_STD'].mean())
    one_hot_df['NEW_CAR_TO_BIRTH_RATIO'] = one_hot_df['OWN_CAR_AGE'] / one_hot_df['DAYS_BIRTH']
    one_hot_df['NEW_CAR_TO_EMPLOY_RATIO'] = one_hot_df['OWN_CAR_AGE'] / one_hot_df['DAYS_EMPLOYED']
    one_hot_df['NEW_PHONE_TO_BIRTH_RATIO'] = one_hot_df['DAYS_LAST_PHONE_CHANGE'] / one_hot_df['DAYS_BIRTH']
    one_hot_df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = one_hot_df['DAYS_LAST_PHONE_CHANGE'] / one_hot_df['DAYS_EMPLOYED']
    one_hot_df['NEW_CREDIT_TO_INCOME_RATIO'] = one_hot_df['AMT_CREDIT'] / one_hot_df['AMT_INCOME_TOTAL']
    
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    one_hot_df= one_hot_df.drop(dropcolum,axis=1)

    one_hot_df, cat_cols = one_hot_encoder(one_hot_df, nan_as_category)

    return one_hot_df

#########################################
# bureau_features
#########################################
def get_bureau_features(nan_as_category = False):
    
    bureau = pd.read_csv('bureau.csv')
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    cat_aggregations = {}
    for cat in bureau_cat: 
        cat_aggregations[cat] = ['mean', 'sum']
        
    bureau['PAYMENT_RATE'] =  bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
    
    agg_dict= {
               #'SK_ID_CURR': ['count'],
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
               'AMT_ANNUITY': ['sum', 'max', 'mean'],
               'PAYMENT_RATE': ['min', 'max', 'sum']
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
        agg_dict[col] = ['mean', 'count', 'sum']
        
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
    
    # removing the duplicate applications
    # Not wokring
    #prev = prev[(prev.NFLAG_LAST_APPL_IN_DAY == 1) & (prev.FLAG_LAST_APPL_PER_CONTRACT == 'Y')]
    
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
    
    prev['PAYMENT_RATE'] = prev['AMT_ANNUITY'] / prev['AMT_CREDIT']
    prev['CREDIT_TO_GOODS_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']


    # Previous applications numeric features
    num_aggregations = {
        'SK_ID_PREV': ['count'],
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'PAYMENT_RATE': ['min', 'max', 'mean'],
        'CREDIT_TO_GOODS_RATIO': ['min', 'max', 'mean']
    }
    
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean', 'sum']

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
    
    pos['SK_DPD_6_months'] = 0 
    pos.loc[pos.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD_6_months'] = pos.loc[pos.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD'] 
    
    pos['SK_DPD_DEF_6_months'] = 0 
    pos.loc[pos.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD_DEF_6_months'] = pos.loc[pos.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD_DEF'] 
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'SK_DPD_6_months': ['max','mean'],
        'SK_DPD_DEF_6_months': ['max', 'mean'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum']
    }
        
    for cat in cat_cols:
        aggregations[cat] = ['mean',  'sum']
        
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg = pos_agg.reset_index()

    pos_dpd_slope = pos.groupby('SK_ID_CURR')['MONTHS_BALANCE', 'SK_DPD'].apply(linear_fit)
    pos_dpd_slope = pos_dpd_slope.reset_index()
    pos_dpd_slope.columns = ['SK_ID_CURR', 'POS_DPD_SLOPE']
    
    pos_agg = pd.merge(pos_agg, pos_dpd_slope, on='SK_ID_CURR', how='left')
    
    pos_dpd_def_slope = pos.groupby('SK_ID_CURR')['MONTHS_BALANCE', 'SK_DPD_DEF'].apply(linear_fit)
    pos_dpd_def_slope = pos_dpd_def_slope.reset_index()
    pos_dpd_def_slope.columns = ['SK_ID_CURR', 'POS_DPD_DEF_SLOPE']
    
    pos_agg = pd.merge(pos_agg, pos_dpd_def_slope, on='SK_ID_CURR', how='left')

    del pos, pos_dpd_slope, pos_dpd_def_slope
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
    
    cc['OVER_DRAFT'] = cc['AMT_TOTAL_RECEIVABLE']/cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['OVER_PAYMENT'] = cc['AMT_PAYMENT_CURRENT']/cc['AMT_INST_MIN_REGULARITY']
    cc['INTEREST'] = cc['AMT_TOTAL_RECEIVABLE']/cc['AMT_RECEIVABLE_PRINCIPAL']
    
    """
    cc['SK_DPD_6_months'] = 0 
    cc.loc[cc.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD_6_months'] = cc.loc[cc.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD'] 
    
    cc['SK_DPD_DEF_6_months'] = 0 
    cc.loc[cc.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD_DEF_6_months'] = cc.loc[cc.MONTHS_BALANCE.isin([-1, -2, -3, -4, -5, -6]), 'SK_DPD_DEF'] 
    """
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    cc_agg = cc_agg.reset_index() 
    
    cc_over_draft_slope = cc.groupby('SK_ID_CURR')['MONTHS_BALANCE', 'OVER_DRAFT'].apply(linear_fit)
    cc_over_draft_slope = cc_over_draft_slope.reset_index()
    cc_over_draft_slope.columns = ['SK_ID_CURR', 'CC_OVER_DRAFT_SLOPE']
    
    cc_agg = pd.merge(cc_agg, cc_over_draft_slope, on='SK_ID_CURR', how='left')

    cc_over_payment_slope = cc.groupby('SK_ID_CURR')['MONTHS_BALANCE', 'OVER_PAYMENT'].apply(linear_fit)
    cc_over_payment_slope = cc_over_payment_slope.reset_index()
    cc_over_payment_slope.columns = ['SK_ID_CURR', 'CC_OVER_PAYMENT_SLOPE']
    
    cc_agg = pd.merge(cc_agg, cc_over_payment_slope, on='SK_ID_CURR', how='left')
  
    cc_interest_slope = cc.groupby('SK_ID_CURR')['MONTHS_BALANCE', 'INTEREST'].apply(linear_fit)
    cc_interest_slope = cc_interest_slope.reset_index()
    cc_interest_slope.columns = ['SK_ID_CURR', 'CC_INTEREST_SLOPE']
    
    cc_agg = pd.merge(cc_agg, cc_interest_slope, on='SK_ID_CURR', how='left')
   
 
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
    
    # number of less payments done
    ins['NLP_Y'] = ins['PAYMENT_DIFF'].apply(lambda x: 1 if x > 0 else 0)
    ins['DPD_Y'] = ins['DPD'].apply(lambda x: 1 if x > 0 else 0)
    ins['DBD_Y'] = ins['DBD'].apply(lambda x: 1 if x > 0 else 0)
    
    ins['DPD_400'] = 0 
    ins.loc[ins.DAYS_INSTALMENT >= -400, 'DPD_400'] = ins.loc[ins.DAYS_INSTALMENT >= -400, 'DPD'] 
       
    ins['DPD_800'] = 0
    ins.loc[ins.DAYS_INSTALMENT >= -800, 'DPD_800'] = ins.loc[ins.DAYS_INSTALMENT >= -800, 'DPD'] 
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'DPD_400': ['max', 'mean', 'sum'],
        'DPD_800': ['max', 'mean', 'sum'],
       # 'NLP_Y': ['max', 'mean', 'sum'],
       # 'DPD_Y': ['max', 'mean', 'sum'],
       # 'DBD_Y': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }

    for cat in cat_cols:
        aggregations[cat] = ['mean', 'sum']
        
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTALL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    ins_agg = ins_agg.reset_index()

    ins_slope = ins[~ins.DAYS_ENTRY_PAYMENT.isnull()].groupby('SK_ID_CURR')['DAYS_ENTRY_PAYMENT', 'DPD'].apply(linear_fit)
    ins_slope = ins_slope.reset_index()
    ins_slope.columns = ['SK_ID_CURR', 'INSTALL_DPD_SLOPE']
    
    ins_agg = pd.merge(ins_agg, ins_slope, on='SK_ID_CURR', how='left')
    
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


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]
        
class LightGBMWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['feature_fraction_seed'] = seed
        params['bagging_seed'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, verbose_eval=True)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf, kf, NFOLDS, X_train, y_train, X_dev, y_dev, application_test):
    ntrain = X_train.shape[0]
    ndev = X_dev.shape[0]
    ntest = application_test.shape[0]

    oof_train = np.zeros((ntrain,))
    oof_dev = np.zeros((ndev,))
    oof_dev_skf = np.empty((NFOLDS, ndev))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        x_tr = X_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = X_train.iloc[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_dev_skf[i, :] = clf.predict(X_dev)
        oof_test_skf[i, :] = clf.predict(application_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_dev[:] = oof_dev_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_dev.reshape(-1, 1), oof_test.reshape(-1, 1)

def train_stacked(whole_df, important_columns, model, version, estimators, desc, algo='GBC'):
    application_train = whole_df[whole_df['TARGET'].notnull()]
    y_true = application_train[['SK_ID_CURR', 'TARGET']]

    application_test = whole_df.loc[whole_df['TARGET'].isnull(), important_columns]
    
    application_train.replace([np.inf, -np.inf], 0, inplace = True)
    application_test.replace([np.inf, -np.inf], 0, inplace = True)
    application_test = application_test.fillna(0)
    
    X_train, y_train, X_dev, y_dev = get_train_dev_data(application_train, y_true, important_columns)

    del application_train, y_true, whole_df
    gc.collect()
        
    NFOLDS = 5
    SEED = 0
    kf = StratifiedKFold(n_splits = NFOLDS, shuffle=True, random_state=SEED)
    
    et_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
    'verbose': 10
    }

    rf_params = {
        'n_jobs': 16,
        'n_estimators': 200,
        'max_features': 0.2,
        'max_depth': 12,
        'min_samples_leaf': 2,
        'verbose': 10

    }

    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.075,
        'objective': 'binary:logistic',
        'max_depth': 4,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'nrounds': 200
    }

    lightgbm_params = {
            'nthread':16,
            'n_estimators':1200,
            'learning_rate':0.02,
            'num_leaves':34,
            'colsample_bytree':0.9497036,
            'subsample':0.8715623,
            'max_depth':8,
            'reg_alpha':0.041545473,
            'reg_lambda':0.0735294,
            'min_split_gain':0.0222415,
            'min_child_weight':39.3259775,
            'verbose':100, 
    }
    
    xg = XgbWrapper(seed=SEED, params=xgb_params)
    et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    lg = LightGBMWrapper(clf = LGBMClassifier, seed = SEED, params = lightgbm_params)

    xg_oof_train, xg_oof_dev, xg_oof_test = get_oof(xg, kf, NFOLDS, X_train, y_train, X_dev, y_dev, application_test)
    et_oof_train, et_oof_dev, et_oof_test = get_oof(et, kf, NFOLDS, X_train, y_train, X_dev, y_dev, application_test)
    rf_oof_train, rf_oof_dev, rf_oof_test = get_oof(rf, kf, NFOLDS, X_train, y_train, X_dev, y_dev, application_test)
    lg_oof_train, lg_oof_dev, lg_oof_test = get_oof(lg, kf, NFOLDS, X_train, y_train, X_dev, y_dev, application_test)

    
    x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, lg_oof_train), axis=1)
    x_dev = np.concatenate((xg_oof_dev, et_oof_dev, rf_oof_dev, lg_oof_dev), axis=1)
    x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, lg_oof_test), axis=1)
    
    print("{},{}".format(x_train.shape, x_test.shape))
    
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train,y_train)
    

    ###################################################
    # results on the train set
    ###################################################
   
    y_train_predict = logistic_regression.predict(x_train)
    y_train_predict_proba = logistic_regression.predict_proba(x_train)
    
    scores_df = get_scores(y_train, y_train_predict, y_train_predict_proba[:, 1], 'train')
             
    ###################################################
    # results on the dev set
    ###################################################
    
    y_dev_predict = logistic_regression.predict(x_dev)
    y_dev_predict_proba = logistic_regression.predict_proba(x_dev)
    
    # get dev scores
    scores_df = scores_df.append(get_scores(y_dev, y_dev_predict, y_dev_predict_proba[:, 1], 'test'))
    
    
    with open('all_reasults.csv', 'a') as f:
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
    
    application_test['TARGET'] = logistic_regression.predict_proba(x_test)[:,1] 
    application_test[['SK_ID_CURR', 'TARGET']].to_csv('first_submission.csv', index=False, float_format='%.8f')

    return 1

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

def LGBM_train_cv(application_train, usecols, params, dropcols, model, version, desc, algo, estimators):
    dtrain = lgb.Dataset(application_train[usecols].drop(dropcols, axis=1), application_train['TARGET'])
    eval_ = lgb.cv(params,
             dtrain,
             nfold=5,
             stratified=True,
             num_boost_round=20000,
             early_stopping_rounds=200,
             metrics='auc',
             verbose_eval=100,
             seed = 5,
             show_stdv=True)
    
    with open('all_reasults.csv', 'a') as f:
        spamwriter = csv.writer(f)
        spamwriter.writerow([model,
                             version,
                             desc,
                             algo,
                             algo+"(n_estimators="+str(estimators)+" random_state=0)",
                             str(round(max(eval_['auc-mean']), 4)),'',
                             '','',
                             '','',
                             "NA","NA","dev set",
                             '','',
                             '','',
                             '',''])
    
    return max(eval_['auc-mean'])
    

def LGBM_train(X_train, y_train, X_dev, y_dev, estimators):
    clf = LGBMClassifier(
            nthread=16,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.5,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.0001,
            reg_lambda=100,
            min_split_gain=0.0222415,
            min_child_weight=50,
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
            nthread=16,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.5,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.0001,
            reg_lambda=100,
            min_split_gain=0.0222415,
            min_child_weight=50,
            random_seed = 42,
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
    
###############################################
# baseline so far
###############################################    
    
whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_installments_payments_features(whole_df, important_columns, 'train')

whole_df['ACTIVE_ANNUITY_TO_INCOME_RATIO'] = whole_df['ACTIVE_AMT_ANNUITY_SUM'] / (1 + whole_df['AMT_INCOME_TOTAL'])
whole_df['ACTIVE_ANNUITY_TO_CREDIT_RATIO'] = whole_df['ACTIVE_AMT_ANNUITY_SUM'] / whole_df['ACTIVE_AMT_CREDIT_SUM_SUM'] 

application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'], application_test
gc.collect()

important_columns = list(application_train.columns)
important_columns.remove('SK_ID_CURR')
important_columns.remove('index')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '0', 10000, 
                                      "LGB trained for ALL the features - baseline", 'LGB')

important_features.sort_values()    

##########################################
# free up memory and run the experiments 
# this is taking too much time
##########################################

whole_df = get_application_train_test()

important_columns = get_base_important_columns()

whole_df, important_columns = add_bureau_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_bureau_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_previous_applications_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_pos_cash_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_credit_card_balance_features(whole_df, important_columns, 'train')
whole_df, important_columns = add_installments_payments_features(whole_df, important_columns, 'train')

whole_df['ACTIVE_ANNUITY_TO_INCOME_RATIO'] = whole_df['ACTIVE_AMT_ANNUITY_SUM'] / (1 + whole_df['AMT_INCOME_TOTAL'])
whole_df['ACTIVE_ANNUITY_TO_CREDIT_RATIO'] = whole_df['ACTIVE_AMT_ANNUITY_SUM'] / whole_df['ACTIVE_AMT_CREDIT_SUM_SUM'] 

application_train = whole_df[whole_df['TARGET'].notnull()]

with open('whole_df', 'wb') as f:
    pickle.dump(whole_df,f)

#with open('whole_df', 'rb') as f:
#    whole_df = pickle.load(f)

del whole_df
gc.collect()


important_columns = list(application_train.columns)
important_columns.remove('SK_ID_CURR')
important_columns.remove('index')

params = {'nthread':8,'learning_rate':0.02, 'num_leaves':34,'colsample_bytree':0.9497036,'subsample':0.8715623,'max_depth':8,'reg_alpha':0.041545473,'reg_lambda':0.0735294,'min_split_gain':0.0222415,'min_child_weight':39.3259775}
LGBM_train_cv(application_train, important_columns, params, [], 'model_15', '1', 
                                      "LGB trained for ALL the features - baseline 5 fold stratified CV", 'LGB', 10000)

##################################################
# handle overfit
#################################################
with open('whole_df', 'rb') as f:
    whole_df = pickle.load(f)
application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'], application_test
gc.collect()

important_columns = list(application_train.columns)
important_columns.remove('SK_ID_CURR')
important_columns.remove('index')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '2', 10000, 
                                      "LGB trained for ALL the features col sample 0.85 reg_lambda- ", 'LGB')

important_features.sort_values()   

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '3', 10000, 
                                      "LGB trained for ALL the features reg_alpha - 0.9- ", 'LGB')

important_features.sort_values()   

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '4', 10000, 
                                      "LGB trained for ALL the features reg_alpha - 0.001- ", 'LGB')

important_features.sort_values()   

# lower alpha more regularization 

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '5', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 0.9", 'LGB')

important_features.sort_values()

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '6', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 2.7", 'LGB')

important_features.sort_values()

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '7', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 10", 'LGB')

important_features.sort_values()

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '8', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 30", 'LGB')

important_features.sort_values()

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '9', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 100", 'LGB')

important_features.sort_values()

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '10', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 500", 'LGB')

important_features.sort_values()

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '10', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 1500", 'LGB')

important_features.sort_values()

#######################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '12', 10000, 
                                      "LGB trained for ALL the features reg_lambda - 100", 'LGB')

important_features.sort_values()

"""
    clf = LGBMClassifier(
            nthread=16,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.0001,
            reg_lambda=100,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )
"""

######################################################
with open('whole_df', 'rb') as f:
    whole_df = pickle.load(f)

feat_importance = kfold_lightgbm(whole_df, num_folds= 5, submission_file_name='submission_23.csv', stratified= False, debug=False)
feat_importance.groupby('feature')['importance'].sum().sort_values()

feat_importance = kfold_lightgbm(whole_df, num_folds= 5, submission_file_name='submission_24.csv', stratified= True, debug=False)
feat_importance.groupby('feature')['importance'].sum().sort_values()


################################################################
with open('whole_df', 'rb') as f:
    whole_df = pickle.load(f)
application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'], application_test
gc.collect()

important_columns = list(application_train.columns)
important_columns.remove('SK_ID_CURR')
important_columns.remove('index')

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '15', 10000, 
                                      "LGB trained for ALL the features col sample - 0.8", 'LGB')

important_features.sort_values()

###############################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '16', 10000, 
                                      "LGB trained for ALL the features col sample - 0.5", 'LGB')

important_features.sort_values()

###############################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '17', 10000, 
                                      "LGB trained for ALL the features col sample - 0.3", 'LGB')

important_features.sort_values()

###############################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '18', 10000, 
                                      "LGB trained for ALL the features col sample - 0.4", 'LGB')

important_features.sort_values()

###############################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '18', 10000, 
                                      "LGB trained for ALL the features col sample - 0.5", 'LGB')

important_features.sort_values()

################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '19', 10000, 
                                      "LGB trained for ALL the features col sample - 0.6", 'LGB')

important_features.sort_values()
"""
        clf = LGBMClassifier(
            nthread=16,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.5,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.0001,
            reg_lambda=100,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            random_seed = 42,
            silent=-1,
            verbose=-1, )
       """ 
       
######################################################
with open('whole_df', 'rb') as f:
    whole_df = pickle.load(f)

feat_importance = kfold_lightgbm(whole_df, num_folds= 5, submission_file_name='submission_25.csv', stratified= False, debug=False)
feat_importance.groupby('feature')['importance'].sum().sort_values()


###########################################################################
with open('whole_df', 'rb') as f:
    whole_df = pickle.load(f)
application_train = whole_df[whole_df['TARGET'].notnull()]
application_test = whole_df[whole_df['TARGET'].isnull()]
y_true = application_train[['SK_ID_CURR', 'TARGET']]

del application_train['TARGET'], application_test
gc.collect()

important_columns = list(application_train.columns)
important_columns.remove('SK_ID_CURR')
important_columns.remove('index')


clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '22', 10000, 
                                      "LGB trained for ALL the features subsabple 0.7- ", 'LGB')

important_features.sort_values()   

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '23', 10000, 
                                      "LGB trained for ALL the features subsample 0.9 ", 'LGB')

important_features.sort_values()   


##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '24', 10000, 
                                      "LGB trained for ALL the features subsample 0.85 ", 'LGB')

important_features.sort_values()   

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '25', 10000, 
                                      "LGB trained for ALL the features subsample 0.8715623 ", 'LGB')

important_features.sort_values()   


##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '26', 10000, 
                                      "LGB trained for ALL the features min child weight 50 ", 'LGB')

important_features.sort_values()  

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '27', 10000, 
                                      "LGB trained for ALL the features min child weight 100 ", 'LGB')

important_features.sort_values()  


##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '28', 10000, 
                                      "LGB trained for ALL the features min child weight 60 ", 'LGB')

important_features.sort_values()  

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '29', 10000, 
                                      "LGB trained for ALL the features min_split_gain 0.04 ", 'LGB')

important_features.sort_values()  

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '30', 10000, 
                                      "LGB trained for ALL the features min_split_gain 0.01 ", 'LGB')

important_features.sort_values()

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '30', 10000, 
                                      "LGB trained for ALL the features num leaves 20 ", 'LGB')

important_features.sort_values()

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '31', 10000, 
                                      "LGB trained for ALL the features num leaves 40 ", 'LGB')

important_features.sort_values()

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '32', 10000, 
                                      "LGB trained for ALL the features num leaves 30 ", 'LGB')

important_features.sort_values()

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '33', 10000, 
                                      "LGB trained for ALL the features lr 0.06 ", 'LGB')

important_features.sort_values()


##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '34', 10000, 
                                      "LGB trained for ALL the features lr 0.0.15 ", 'LGB')

important_features.sort_values()

##################################################################
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '34', 10000, 
                                      "LGB trained for ALL the features lr 0.1", 'LGB')

important_features.sort_values()

#################################################################
with open('whole_df', 'rb') as f:
    whole_df = pickle.load(f)

feat_importance = kfold_lightgbm(whole_df, num_folds= 5, submission_file_name='submission_26.csv', stratified= False, debug=False)
feat_importance.groupby('feature')['importance'].sum().sort_values()

feat_importance = kfold_lightgbm(whole_df, num_folds= 5, submission_file_name='submission_27.csv', stratified= True, debug=False)
feat_importance.groupby('feature')['importance'].sum().sort_values()

################################################################

clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '38', 10000, 
                                      "LGB trained for ALL the features", 'LGB')

important_features.sort_values()

important_features = list(important_features[important_features > 0.04].index)

clf, important_features = train_model(application_train, y_true, important_features, 'model_15', '39', 10000, 
                                      "LGB trained for ALL the features less features", 'LGB')


############################################################

important_columns.remove('NEW_SOURCES_PROD')
clf, important_features = train_model(application_train, y_true, important_columns, 'model_15', '40', 10000, 
                                      "LGB trained for ALL the features", 'LGB')
