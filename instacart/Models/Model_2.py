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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



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

# Products information ----------------------------------------------------------------
# add order information to priors set
priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

# create new variables


asile_details = pd.merge(priors, products, on='product_id')

agg_dict_2 = {'reordered':{'_aisle_re_sum':'sum', 
                           '_aisle_re_mean': 'mean'}}
ailse = ka_add_groupby_features_1_vs_n(asile_details, ['aisle_id'], agg_dict_2)

agg_dict_2 = {'reordered':{'_dept_re_sum':'sum', 
                           '_dept_re_mean': 'mean'}}
dept = ka_add_groupby_features_1_vs_n(asile_details, ['department_id'], agg_dict_2)

## proudct asile dept
ailse =  pd.merge(products, ailse, on='aisle_id', how='left')
ailse_dept =  pd.merge(ailse, dept, on='department_id', how='left')

ailse_dept = ailse_dept.drop(['product_name', 'aisle_id', 'department_id'], axis=1)
del asile_details, ailse, dept
gc.collect()


# _user_buy_product_times: 

##------ is it ordered by time otherwise cumcount will give wrong results
priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1
# _prod_tot_cnts:
# _reorder_tot_cnts_of_this_prod: 
# _prod_order_once: 
# _prod_order_more_than_once: 
agg_dict = {'user_id':{'_prod_tot_cnts':'count'}, 
            'reordered':{'_prod_reorder_tot_cnts':'sum'}, 
            'add_to_cart_order':{'_prod_add_to_cart_mean':'mean', 
                                 '_prod_add_to_cart_median':'median',
                                 '_prod_add_to_cart_sum':'sum',
                                 '_prod_add_to_cart_min':'min',
                                 '_prod_add_to_cart_max':'max'}, 
            '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                        '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)}}
prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

# _prod_reorder_prob:
# _prod_reorder_ratio: 
prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt
   
   


# _user_total_orders:
# _user_sum_days_since_prior_order: ，priors_orders_detail order levelunique
# _user_mean_days_since_prior_order: 
agg_dict_2 = {'order_number':{'_user_total_orders':'max'},
              'days_since_prior_order':{'_user_sum_days_since_prior_order':'sum', 
                                        '_user_mean_days_since_prior_order': 'mean',
                                        '_user_median_days_since_prior_order': 'median',
                                        '_user_min_days_since_prior_order': 'min',
                                        '_user_max_days_since_prior_order': 'max',
                                        '_user_scheduled_weekly_orders':lambda x: sum(x==7)+sum(x==14)+sum(x==21)+sum(x==28)+sum(x==15)}, 
                'order_dow':{'_user_dow_mean':'mean', '_user_dow_sum':'sum'},
                'order_hour_of_day':{'_user_hour_of_day_mean':'mean',
                                     '_user_hour_of_day_sum':'sum'}}

users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'], agg_dict_2)
users['_user_scheduled_orders_ratio'] = users._user_scheduled_weekly_orders/users._user_total_orders
# _user_reorder_ratio: reorder
# _user_total_products: 
# _user_distinct_products: 
'''
agg_dict_3 = {'reordered':
              {'_user_reorder_ratio': 
               lambda x: sum(priors_orders_detail.ix[x.index,'reordered']==1)/
                         sum(priors_orders_detail.ix[x.index,'order_number'] > 1)},
              'product__prod_add_to_cart_meanid':{'_user_total_products':'count', 
                            '_user_distinct_products': lambda x: x.nunique()}}
us = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['user_id'], agg_dict_3)
'''

us = pd.concat([
    priors_orders_detail.groupby('user_id')['product_id'].count().rename('_user_total_products'),
    priors_orders_detail.groupby('user_id')['product_id'].nunique().rename('_user_distinct_products'),
    (priors_orders_detail.groupby('user_id')['reordered'].sum() /
        priors_orders_detail[priors_orders_detail['order_number'] > 1].groupby('user_id')['order_number'].count()).rename('_user_reorder_ratio')
], axis=1).reset_index()

users = users.merge(us, how='inner')


users['_user_average_basket'] = users._user_total_products / users._user_total_orders

us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

users = users.merge(us, how='inner')

# _up_order_count:
# _up_first_order_number:
# _up_last_order_number: 
# _up_average_cart_position:
agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                              '_up_first_order_number': 'min', 
                              '_up_last_order_number':'max'}, 
              'add_to_cart_order':{'_up_average_cart_position': 'mean'}}

data = ka_add_groupby_features_1_vs_n(df=priors_orders_detail, 
                                                      group_columns_list=['user_id', 'product_id'], 
                                                      agg_dict=agg_dict_4)

data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')

data['_up_order_rate'] = data._up_order_count / data._user_total_orders
data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders - data._up_first_order_number + 1)

# add user_id to train set
train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')

data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')
data = data.merge(right=ailse_dept, how='left', on='product_id')
# train and test set
train = data.loc[data.eval_set == "train",:]
train.loc[:, 'reordered'] = train.reordered.fillna(0)

X_test = data.loc[data.eval_set == "test",:]

# release Memory
# del train, prd, users
# gc.collect()
# release Memory
del priors_orders_detail, orders
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
kf = GroupKFold(n_splits=10)
for i, (train_index, val_index) in enumerate(kf.split(train, groups=train['user_id'].values)):
    X_train, X_val = train.iloc[train_index], train.iloc[val_index]
    y_train, y_val = train.iloc[train_index].reordered, train.iloc[val_index].reordered

    d_train = xgboost.DMatrix(X_train.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1), y_train)
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
    
    ## get the f1 score on x_val and average it across 10 folds 
    d_val = xgboost.DMatrix(X_val.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1))
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
d_train = xgboost.DMatrix(train.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1), train.reordered)
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

d_test = xgboost.DMatrix(X_test.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1))
X_test.loc[:,'reordered'] = (bst.predict(d_test) > 0.21).astype(int)
X_test.loc[:, 'product_id'] = X_test.product_id.astype(str)
submit = ka_add_groupby_features_n_vs_1(X_test[X_test.reordered == 1], 
                                               group_columns_list=['order_id'],
                                               target_columns_list= ['product_id'],
                                               methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
submit.columns = sample_submission.columns.tolist()
submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
submit_final.to_csv("python_test.csv", index=False)


