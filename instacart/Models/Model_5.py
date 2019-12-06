#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:31:58 2017

@author: neha
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import gc
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#none handeling

def load_data(path_data):
    '''
    --------------------------------order_product--------------------------------
    * Unique in order_id + product_id
    '''
    priors = pd.read_csv(path_data + 'order_products__prior.csv', 
                     dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    train = pd.read_csv(path_data + 'order_products__train.csv', 
                    dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    '''
    --------------------------------order--------------------------------
    * This file tells us which set (prior, train, test) an order belongs
    * Unique in order_id
    * order_id in train, prior, test has no intersection
    * this is the #order_number order of this user
    '''
    orders = pd.read_csv(path_data + 'orders.csv', 
                         dtype={
                                'order_id': np.int32,
                                'user_id': np.int64,
                                'eval_set': 'category',
                                'order_number': np.int16,
                                'order_dow': np.int8,
                                'order_hour_of_day': np.int8,
                                'days_since_prior_order': np.float32})

    #  order in prior, train, test has no duplicate
    #  order_ids_pri = priors.order_id.unique()
    #  order_ids_trn = train.order_id.unique()
    #  order_ids_tst = orders[orders.eval_set == 'test']['order_id'].unique()
    #  print(set(order_ids_pri).intersection(set(order_ids_trn)))
    #  print(set(order_ids_pri).intersection(set(order_ids_tst)))
    #  print(set(order_ids_trn).intersection(set(order_ids_tst)))

    '''
    --------------------------------product--------------------------------
    * Unique in product_id
    '''
    products = pd.read_csv(path_data + 'products.csv')
    aisles = pd.read_csv(path_data + "aisles.csv")
    departments = pd.read_csv(path_data + "departments.csv")
    sample_submission = pd.read_csv(path_data + "sample_submission.csv")
    
    return priors, train, orders, products, aisles, departments, sample_submission

class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ......")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))
            
def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with tick_tock("add stats features"):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(group_columns_list + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if only_new_feature:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new

def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_stats_features_n_vs_1(train, group_columns_list=['x0'], target_columns_list=['x10'])
    '''
    with tick_tock("add stats features", verbose):
        dicts = {"group_columns_list": group_columns_list , "target_columns_list": target_columns_list, "methods_list" :methods_list}

        for k, v in dicts.items():
            try:
                if type(v) == list:
                    pass
                else:
                    raise TypeError(k + "should be a list")
            except TypeError as e:
                print(e)
                raise

        grouped_name = ''.join(group_columns_list)
        target_name = ''.join(target_columns_list)
        combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped[target_name].agg(methods_list).reset_index()
        the_stats.columns = [grouped_name] + \
                            ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                             for (grouped_name, method_name, target_name) in combine_name]
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new



path_data = '../data/'
priors, train, orders, products, aisles, departments, sample_submission = load_data(path_data)

#insert none o products, aisles and departments
print(aisles.aisle_id.max())
print(departments.department_id.max())
print(products.product_id.max())

aisles.loc[135] = [135, 'None']
departments.loc[22] = [22, 'None']
products.loc[49689] = [49689, 'None', 135, 22]

#insert none product to priors and train
prior_none = priors.merge(orders[['order_id', 'order_number']], on='order_id', how='inner')

none_orders = prior_none[prior_none.order_number > 1].groupby('order_id').agg({'reordered': 'sum', 'add_to_cart_order':'max'}).reset_index()
none_orders = none_orders[none_orders.reordered == 0]
none_orders.loc[:, 'add_to_cart_order'] = none_orders.loc[:, 'add_to_cart_order'] + 1
none_orders.loc[:, 'product_id'] = 49689

none_orders = none_orders.merge(orders[['order_id', 'user_id']], on='order_id', how='inner' )
none_orders['_times'] =  none_orders.groupby('user_id').cumcount()             
none_orders['reordered'] =  (none_orders['_times'] >= 1).astype(int)            

priors =priors.append(none_orders[['order_id', 'product_id', 'add_to_cart_order', 'reordered']], ignore_index=True)

none_orders_tr = train.groupby('order_id').agg({'reordered': 'sum', 'add_to_cart_order':'max'}).reset_index()
none_orders_tr = none_orders_tr[none_orders_tr.reordered == 0]
none_orders_tr.loc[:, 'add_to_cart_order'] = none_orders_tr.loc[:, 'add_to_cart_order'] + 1
none_orders_tr.loc[:, 'product_id'] = 49689
none_orders_tr = none_orders_tr.merge(orders[['order_id', 'user_id']], on='order_id', how='inner' )

none_orders_tr.loc[none_orders_tr.user_id.isin(none_orders.user_id), 'reordered'] = 1

train =train.append(none_orders_tr[['order_id', 'product_id', 'add_to_cart_order', 'reordered']], ignore_index=True)

                   
del prior_none, none_orders, none_orders_tr              
gc.collect()    
                   
                   
orders['_user_days_since_order_cumsum'] = orders.groupby('user_id').days_since_prior_order.cumsum()

# Products information ----------------------------------------------------------------
# add order information to priors set
priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')
priors_orders_detail = priors_orders_detail.merge(products, how='inner', on='product_id')
priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

# create new variables

'''
    User Features: #Products purchased, #Orders made, frequency and recency of orders, #Aisle purchased from, #Department purchased from, frequency and recency of reorders, tenure, mean order size, etc.

    Product Features: #users, #orders, order frequency, reorder rate, recency, mean add_to_cart_order, etc.

    Aisle and Department Features: similar to product features

    user product interaction:#purchases, #reorders, #day since last purchase, #order since last purchase etc.

    User aisle and department interaction: similar to product interaction

    User time interaction: user preferred day of week, user preferred time of day, similar features for products and aisles
'''

users = pd.concat([priors_orders_detail.groupby('user_id').product_id.count().rename('_user_total_products_purchased'), 
                   priors_orders_detail.groupby('user_id').product_id.nunique().rename('_user_total_unique_products'),
                   priors_orders_detail[priors_orders_detail.reordered == 1].groupby('user_id').product_id.nunique().rename('_user_total_unique_products_reordered'),
                   priors_orders_detail[priors_orders_detail.reordered == 1].groupby('user_id').order_id.nunique().rename('_user_total_reorders_made'),
                   priors_orders_detail.groupby('user_id').order_id.nunique().rename('_user_total_orders_made'),
                   priors_orders_detail.groupby('user_id').reordered.sum().rename('_user_total_reorderd_products'),
                   priors_orders_detail[priors_orders_detail['order_number'] > 1].groupby('user_id').order_id.count().rename('_user_total_pd_in_repeat_orders'),
                   priors_orders_detail.groupby('user_id').aisle_id.nunique().rename('_user_total_unique_aisle'),
                   priors_orders_detail.groupby('user_id').department_id.nunique().rename('_user_total_unique_dept')], 
         axis=1).reset_index()

users['_user_mean_order_size'] = users._user_total_products_purchased/users._user_total_orders_made
users['_user_reorded_products_per_order'] = users._user_total_reorderd_products/users._user_total_orders_made
users['_user_reorder_ratio'] = users._user_total_reorderd_products/users._user_total_pd_in_repeat_orders
users['_user_reorder_product_ratio'] = users._user_total_unique_products_reordered/users._user_total_unique_products
users['_percent_orders_with_reorders'] = users._user_total_reorders_made/users._user_total_orders_made
     
     
priors_orders_detail = pd.merge(priors_orders_detail, users[['user_id', '_user_total_orders_made']], on='user_id', how='left')


us_re = pd.concat([priors_orders_detail[priors_orders_detail['order_number'] > priors_orders_detail['_user_total_orders_made'] -2].groupby('user_id').product_id.count().rename('_user_recent_product'),
        priors_orders_detail[priors_orders_detail['order_number'] > priors_orders_detail['_user_total_orders_made'] - 2].groupby('user_id').reordered.sum().rename('_user_recent_reorders'),
        priors_orders_detail[priors_orders_detail['order_number'] > priors_orders_detail['_user_total_orders_made'] -2].groupby('user_id').product_id.nunique().rename('_user_recent_unique_product'),
        priors_orders_detail[(priors_orders_detail['order_number'] > priors_orders_detail['_user_total_orders_made'] - 2) & (priors_orders_detail.reordered == 1)].groupby('user_id').product_id.nunique().rename('_user_recent_unique_product_reordered')
  
    ], axis=1).reset_index()

us_re['_user_recent_reorder_ratio'] = us_re._user_recent_reorders/us_re._user_recent_product
us_re['_user_recent_reorder_product_ratio'] = us_re._user_recent_unique_product/us_re._user_recent_unique_product_reordered

users = pd.merge(users, us_re, on='user_id', how='left')
users['_user_reorder_trend'] = users._user_recent_reorder_ratio/users._user_reorder_ratio
users['_user_pd_reorder_trend'] = users._user_recent_reorder_product_ratio/users._user_reorder_product_ratio


us = pd.concat([orders[orders.eval_set == 'prior'].groupby('user_id').days_since_prior_order.sum().rename('_user_tenure'), 
                orders[orders.eval_set == 'prior'].groupby('user_id').days_since_prior_order.mean().rename('_user_order_in_avg_days'),
                orders[orders.eval_set == 'prior'].groupby('user_id').order_dow.agg(lambda x: x.value_counts().index[0]).rename('_user_dow_pref'),
                orders[orders.eval_set == 'prior'].groupby('user_id').order_hour_of_day.agg(lambda x: x.value_counts().index[0]).rename('_user_hour_pref')
                ], 
                axis=1).reset_index()

users = pd.merge(users, us, on='user_id', how='left')

us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)
users = users.merge(us, how='inner')

del us, us_re
gc.collect()

# asile dept details
asile = pd.concat([priors_orders_detail.groupby('aisle_id').reordered.count().rename('_aisle_total_orders'), 
                   priors_orders_detail.groupby('aisle_id').reordered.sum().rename('_aisle_reorderd_products')],axis=1).reset_index()

dept = pd.concat([priors_orders_detail.groupby('department_id').reordered.count().rename('_dept_total_products'), 
                   priors_orders_detail.groupby('department_id').reordered.sum().rename('_dept_reorderd_products')],axis=1).reset_index()

asile['_aisle_reorder_rate'] = asile._aisle_reorderd_products/asile._aisle_total_orders
dept['_dept_reorder_rate'] = dept._dept_reorderd_products / dept._dept_total_products
ailse =  pd.merge(products[['product_id', 'aisle_id', 'department_id']], asile, on='aisle_id', how='left')
ailse_dept =  pd.merge(ailse, dept, on='department_id', how='left')


# product features Product Features: #users, #orders, order frequency, reorder rate, recency, mean add_to_cart_order, etc.

prod = pd.concat([priors_orders_detail.groupby('product_id').order_id.count().rename('_prod_total_orders'),
                  priors_orders_detail.groupby('product_id').reordered.sum().rename('_prod_total_reorder'),
                  priors_orders_detail.groupby('product_id').add_to_cart_order.mean().rename('_prod_add_to_cart_mean'),
                  priors_orders_detail.groupby('product_id')._user_buy_product_times.agg(lambda x: sum(x==1)).rename('_prod_buy_first_time_total_cnt'),
                  priors_orders_detail.groupby('product_id')._user_buy_product_times.agg(lambda x: sum(x==2)).rename('_prod_buy_second_time_total_cnt'),
                  priors_orders_detail.groupby('product_id')._user_buy_product_times.max().rename('_prod_buy_max_time'),
                  priors_orders_detail.groupby('product_id')._user_buy_product_times.min().rename('_prod_buy_min_time'),
                  priors_orders_detail.groupby('product_id')._user_buy_product_times.agg(lambda x: x.value_counts().index[0]).rename('_prod_buy_time_pref')
                  ], axis=1).reset_index()

prod['_prod_reorder_prob'] = prod._prod_buy_second_time_total_cnt / prod._prod_buy_first_time_total_cnt
prod['_prod_reorder_ratio'] = prod._prod_total_reorder / prod._prod_total_orders
prod['_prod_reorder_times'] = 1 + prod._prod_total_reorder / prod._prod_buy_first_time_total_cnt
 
prod = pd.merge(prod, ailse_dept, how='inner', on='product_id')

del ailse, dept, asile, ailse_dept
gc.collect()
## user_product interaction
# user product interaction:#purchases, #reorders, #day since last purchase, #order since last purchase etc.

data = pd.concat([priors_orders_detail.groupby(['user_id', 'product_id']).order_id.count().rename('_up_order_count'),
                  priors_orders_detail.groupby(['user_id', 'product_id']).reordered.sum().rename('_up_order_reordered_count'),
                  priors_orders_detail[priors_orders_detail['order_number'] > priors_orders_detail['_user_total_orders_made'] - 2].groupby(['user_id', 'product_id']).reordered.sum().rename('_up_order_recent_reordered_count'),
                  priors_orders_detail.groupby(['user_id', 'product_id']).order_number.min().rename('_up_first_order_number'),
                  priors_orders_detail.groupby(['user_id', 'product_id']).order_number.max().rename('_up_last_order_number'),
                  priors_orders_detail.groupby(['user_id', 'product_id'])._user_days_since_order_cumsum.max().rename('_up_days_since_last_order'),
                  priors_orders_detail.groupby(['user_id', 'product_id']).add_to_cart_order.mean().rename('_up_average_cart_position')
        ], axis=1).reset_index()
'''
data_reorder = pd.concat([priors_orders_detail[priors_orders_detail.reordered == 1].groupby(['user_id', 'product_id']).order_id.count().rename('_up_order_count'),
                  priors_orders_detail[priors_orders_detail.reordered == 1].groupby(['user_id', 'product_id']).order_number.min().rename('_up_first_order_number'),
                  priors_orders_detail[priors_orders_detail.reordered == 1].groupby(['user_id', 'product_id']).order_number.max().rename('_up_last_order_number'),
                  priors_orders_detail[priors_orders_detail.reordered == 1].groupby(['user_id', 'product_id']).add_to_cart_order.mean().rename('_up_average_cart_position')
        ], axis=1).reset_index()
'''

data = data.merge(prod, how='inner', on='product_id').merge(users, how='inner', on='user_id')

data['_up_order_rate'] = data._up_order_count / data._user_total_orders_made
data['_up_product_rate'] = data._up_order_count / data._user_total_products_purchased
data['_up_order_since_last_order'] = data._user_total_orders_made - data._up_last_order_number
data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders_made - data._up_first_order_number + 1)
data['_up_days_since_last_order'] = data['_user_tenure'] - data['_up_days_since_last_order']
data['_up_days_since_last_order_percent'] = data['_up_days_since_last_order']/data['_user_tenure']


# add user_id to train set
train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')

data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

# train and test set
train = data.loc[data.eval_set == "train",:]
train.loc[:, 'reordered'] = train.reordered.fillna(0)

X_test = data.loc[data.eval_set == "test",:]

# release Memory
# del train, prd, users
# gc.collect()
# release Memory
del priors_orders_detail, orders, priors, data
gc.collect()

## train on whole data set and predict results 

## TO DO write a CV function 

# F1 score calculation
def f1_score_single(y_true, y_pred):
    y_true = y_true.split()
    y_pred = y_pred.split()
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)
    
def f1_score(y_true, y_pred):
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])

# divide the data into group k-folds
import xgboost
from sklearn.model_selection import GroupKFold

f1=[]
kf = GroupKFold(n_splits=5)
for i, (train_index, val_index) in enumerate(kf.split(train, groups=train['user_id'].values)):
    X_train, X_val = train.iloc[train_index], train.iloc[val_index]
    y_train, y_val = train.iloc[train_index].reordered, train.iloc[val_index].reordered

    d_train = xgboost.DMatrix(X_train.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id', 'aisle_id', 'department_id'], axis=1), y_train)
    d_train = lgb.Dataset(X_train.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id', 'aisle_id', 'department_id'], axis=1),
                      label=y_train)  # , 'order_hour_of_day', 'dow'

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 96,
        'max_depth': 10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5
    }
    ROUNDS = 100
    
    #bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
    #xgboost.plot_importance(bst)
    
    bst = lgb.train(params, d_train, ROUNDS)

    ## get the f1 score on x_val and average it across 10 folds 
    #d_val = xgboost.DMatrix(X_val.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id', 'aisle_id', 'department_id'], axis=1))
    d_val = X_val.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id', 'aisle_id', 'department_id'], axis=1)
    X_val.loc[:,'reordered_prob'] = (bst.predict(d_val)).astype(float)
    X_val.loc[:,'reordered_pred'] = (bst.predict(d_val) > 0.21).astype(int)
    #X_val.ix[X_val['product_id'] == '24852','reordered_pred'] = (X_val[X_val.product_id == '24852'].reordered_prob > 0.21).astype(int)

    X_val.loc[:, 'product_id'] = X_val.product_id.astype(str)
    y_pred = ka_add_groupby_features_n_vs_1(X_val[X_val.reordered_pred == 1], 
                                               group_columns_list=['order_id'],
                                               target_columns_list= ['product_id'],
                                               methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
    y_true = ka_add_groupby_features_n_vs_1(X_val[X_val.reordered == 1], 
                                               group_columns_list=['order_id'],
                                               target_columns_list= ['product_id'],
                                               methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
    val_uniq = pd.DataFrame(X_val.order_id.unique(), columns=['order_id'])
    y_pred = val_uniq.merge(y_pred, how='left').fillna('None')
    y_true = val_uniq.merge(y_true, how='left').fillna('None')
    y_pred.columns = ['order_id', 'products']
    y_true.columns = ['order_id', 'products']
    y_pred = y_pred.sort_values(by=['order_id'])
    y_true = y_true.sort_values(by=['order_id'])
    #print(y_pred.head())
    #print(y_pred.shape[0])
    #print(y_true.head())
    #print(y_true.shape[0])
    f1.append(f1_score(y_true.products.values, y_pred.products.values))
    print(f1_score(y_true.products.values, y_pred.products.values))
    ''' 
    fpr, tpr, thresholds = roc_curve(X_val[X_val.product_id == '24852'].reordered, X_val[X_val.product_id == '24852'].reordered_prob)
    roc_auc = auc(fpr, tpr) # compute area under the curve
     
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
     
    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])
     
    plt.savefig('roc_and_threshold.png')
    plt.close()
    '''
print("mean f1 score")
print(np.mean(f1))



# train on all of the data and submit the results
d_train = lgb.Dataset(train.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id', 'aisle_id', 'department_id'], axis=1),
                      label=train.reordered)  # , 'order_hour_of_day', 'dow'

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 100

bst = lgb.train(params, d_train, ROUNDS)

d_test = X_test.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id', 'aisle_id', 'department_id'], axis=1)
X_test.loc[:,'reordered'] = (bst.predict(d_test) > 0.21).astype(int)
X_test.loc[:, 'product_id'] = X_test.product_id.astype(str)
X_test.loc[X_test['product_id'] == '49689', 'product_id'] = 'None'

submit = ka_add_groupby_features_n_vs_1(X_test[X_test.reordered == 1], 
                                               group_columns_list=['order_id'],
                                               target_columns_list= ['product_id'],
                                               methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
submit.columns = sample_submission.columns.tolist()
submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
submit_final.to_csv("python_test.csv", index=False)

## just for feature importance 

d_train = xgboost.DMatrix(train.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id', 'aisle_id', 'department_id'], axis=1), train.reordered)
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(d_train, "train")]
bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
xgboost.plot_importance(bst)

