#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:25:12 2018

@author: neha

seprate out dev, and eyeball dev set

"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, confusion_matrix
import lightgbm as lgb
import random
import pickle


application_train = pd.read_csv('application_train.csv') # (307511, 122)
application_test = pd.read_csv('application_test.csv') # (48744, 121)

y_true_skid = list(application_train.loc[application_train['TARGET'] == 1, 'SK_ID_CURR'])
y_false_skid = list(application_train.loc[application_train['TARGET'] == 0, 'SK_ID_CURR'])


# get random 5k as dev set calss 1 and 50k calss 2
# out of this 500 of each class as eyeball set
random.seed(42)

cv_y_true = random.sample(y_true_skid, 5000)
cv_y_false = random.sample(y_false_skid, 50000)

train_y_true = list(set(y_true_skid) - set(cv_y_true))
train_y_false = list(set(y_false_skid) - set(cv_y_false))

cv_y_true_eyeball = random.sample(cv_y_true, 500)
cv_y_false_eyeball = random.sample(cv_y_false, 500)

dev_all = cv_y_true+cv_y_false
train_all = train_y_true+train_y_false
dev_eyeball = cv_y_true_eyeball+cv_y_false_eyeball

with open("train_all.txt", "wb") as fp:
    pickle.dump(train_all, fp)
    
with open("dev_all.txt", "wb") as fp:
    pickle.dump(dev_all, fp)

with open("dev_eyeball.txt", "wb") as fp:
    pickle.dump(dev_eyeball, fp)



######## load the files into variables for CV and analysis ##################
with open("train_all.txt", "rb") as fp:
    train_all_skid = pickle.load(fp)
    
with open("dev_all.txt", "rb") as fp:
    dev_all_skid = pickle.load(fp)
    
with open("dev_eyeball.txt", "rb") as fp:
    dev_eyeball_skid = pickle.load(fp)
    
    
application_train_data = application_train[application_train.SK_ID_CURR.isin(train_all_skid)]  
application_dev_data = application_train[application_train.SK_ID_CURR.isin(dev_all_skid)]
application_dev_eyeball_data = application_train[application_train.SK_ID_CURR.isin(dev_eyeball_skid)]