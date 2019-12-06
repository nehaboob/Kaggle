from sklearn.model_selection import StratifiedKFold, KFold
import csv
import pandas as pd
import pickle
import click
import logging
import gc
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import datetime
import gc

def add_date_colums(df):
    df['purchase_month'] = df['purchase_date'].dt.month
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['month_diff'] = ((datetime.datetime.strptime('2018-04-30 23:59:59', '%Y-%m-%d %H:%M:%S') - df['purchase_date']).dt.days)//30
    return df

def handel_categorical(df):
	# convert caterorical variables to one hot encoded

	for col in ['authorized_flag', 'category_1']:
		df[col] = df[col].map({'Y':1, 'N':0})
    
	df = pd.get_dummies(df, columns=['category_2', 'category_3'])

	cat_feat = [col for col in df.columns if col.startswith(('authorized_flag', 'category_1', 'category_2', 'category_3'))]

	return df, cat_feat

def get_merch_features(df):
    print("start merchants agg")
    merch = pd.read_csv('./data/raw/merchants.csv')
    merch = merch.drop_duplicates(subset='merchant_id')
    merch['category_4'] = merch['category_4'].map({'Y':1, 'N':0})
    merch['most_recent_purchases_range'] = merch['most_recent_purchases_range'].map({'A':5, 'B':4, 'C':3, 'D':2, 'E':1}) 
    merch['most_recent_sales_range'] = merch['most_recent_sales_range'].map({'A':5, 'B':4, 'C':3, 'D':2, 'E':1})  
    merch.columns = ['merch_'+c if c != 'merchant_id' else c for c in merch.columns]
    df = pd.merge(df, merch, on='merchant_id', how='left')
    del merch
    gc.collect()

    aggs = {'merch_active_months_lag12': ['sum', 'mean', 'max', 'min', 'std'],
        'merch_active_months_lag3':['sum', 'mean', 'max', 'min', 'std'],  
        'merch_active_months_lag6':['sum', 'mean', 'max', 'min', 'std'],  
        'merch_avg_purchases_lag12':['sum', 'mean', 'max', 'min', 'std'], 
        'merch_avg_purchases_lag3':['sum', 'mean', 'max', 'min', 'std'],  
        'merch_avg_purchases_lag6':['sum', 'mean', 'max', 'min', 'std'],  
        'merch_avg_sales_lag12':['sum', 'mean', 'max', 'min', 'std'], 
        'merch_avg_sales_lag3':['sum', 'mean', 'max', 'min', 'std'], 
        'merch_avg_sales_lag6':['sum', 'mean', 'max', 'min', 'std'],
        'merch_most_recent_purchases_range':['sum', 'mean', 'max', 'min', 'std'],
        'merch_most_recent_sales_range':['sum', 'mean', 'max', 'min', 'std'],
        'merch_numerical_1':['sum', 'mean', 'max', 'min', 'std'],
        'merch_numerical_2':['sum', 'mean', 'max', 'min', 'std'],
        'merch_category_4':['mean', 'sum'],
        'merch_merchant_group_id':['nunique']                                 
    }

    transactions = df.groupby('card_id').agg(aggs)
    transactions.columns = ['_'.join(col).strip() for col in transactions.columns.values]
    transactions = transactions.reset_index()
    print("merchants_agg done")
    return transactions

def aggregate_transactions(df, cat_feats=[]):
    print('start aggregate_transactions')    
    df.loc[:, 'purchase_date_ptp'] = pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9
    aggs = {
        'card_id': ['size'],
        'month':['nunique'],
        'hour':['nunique'],
        'weekofyear':['nunique'],
        'dayofweek':['nunique'],
        'year':['nunique'],
        'month_diff': ['mean', 'nunique'],
        'weekend': ['sum', 'mean'],
 		'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date_ptp': [np.ptp],
        'purchase_date': ['min', 'max'],
        'month_lag': ['min', 'max', 'mean', 'std', 'nunique']
        }

    for f in cat_feats: 
        aggs[f] = ['mean', 'sum']

    transactions = df.groupby('card_id').agg(aggs)   
    transactions.columns = ['_'.join(col).strip() for col in transactions.columns.values]
    transactions = transactions.reset_index() 
    transactions['purchase_date_diff'] = (transactions['purchase_date_max'] - transactions['purchase_date_min']).dt.days
    transactions['purchase_date_average'] = transactions['purchase_date_diff']/transactions['card_id_size']
    transactions['purchase_date_uptonow'] = (datetime.datetime.strptime('2018-04-30 23:59:59', '%Y-%m-%d %H:%M:%S') - transactions['purchase_date_max']).dt.days
    transactions['merch_tran_ratio'] = transactions['card_id_size']/transactions['merchant_id_nunique']
    transactions['subsect_tran_ratio'] = transactions['card_id_size']/transactions['subsector_id_nunique']
    transactions['merch_subsect_ratio'] = transactions['merchant_id_nunique']/transactions['subsector_id_nunique']
    transactions['merch_purchase_sum_ratio'] = transactions['purchase_amount_sum']/transactions['merchant_id_nunique']
    transactions['merch_purchase_mean_ratio'] = transactions['purchase_amount_mean']/transactions['merchant_id_nunique']
    transactions['subsect_purchase_sum_ratio'] = transactions['purchase_amount_sum']/transactions['subsector_id_nunique']
    transactions['subsect_purchase_mean_ratio'] = transactions['purchase_amount_mean']/transactions['subsector_id_nunique']
    transactions['merch_month_diff'] = transactions['merchant_id_nunique']/transactions['month_lag_nunique']
    transactions['purchase_sum_month_diff'] = transactions['purchase_amount_sum']/transactions['month_lag_nunique']
    transactions['purchase_mean_month_diff'] = transactions['purchase_amount_mean']/transactions['month_lag_nunique']

    
    aggs = {
        'merchant_id': ['nunique', 'count'],
        'merchant_category_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std']                               
    }
    transactions_by_lag = df.groupby(['card_id', 'month_lag']).agg(aggs)
    transactions_by_lag.columns = ['_'.join(col).strip() for col in transactions_by_lag.columns.values]
    transactions_by_lag = transactions_by_lag.reset_index()
    transactions_by_lag = transactions_by_lag.pivot(index='card_id', columns='month_lag')
    transactions_by_lag.columns =[(col[0]+'_'+str(col[1])).strip() for col in transactions_by_lag.columns.values]
    transactions_by_lag = transactions_by_lag.reset_index()

    transactions = pd.merge(transactions, transactions_by_lag, on='card_id', how='inner')    
    
    print('User aggregation done')

    return transactions
	
def get_transaction_features(df):
	df = add_date_colums(df)
	df, cat_feats = handel_categorical(df)
	print(cat_feats)
	df = aggregate_transactions(df, cat_feats)
	return df

def get_final_features(df):
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.strptime('2018-04-30 23:59:59', '%Y-%m-%d %H:%M:%S') - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_purchase_date_max','new_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
    df['card_id_total'] = df['new_card_id_size']+df['hist_card_id_size']
    df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']

    return df

@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
	
    # get hist features
    historical_transactions = pd.read_csv('./data/raw/historical_transactions.csv', parse_dates=['purchase_date'])
    hist_merch_agg = get_merch_features(historical_transactions[['card_id', 'merchant_id']])
    historical_transactions = get_transaction_features(historical_transactions)
    historical_transactions = pd.merge(historical_transactions, hist_merch_agg, on='card_id', how='left')
    historical_transactions.columns = ['hist_' + c if c != 'card_id' else c for c in historical_transactions.columns]
    historical_transactions.to_csv('./data/processed/historical_transactions_agg_v8.csv')

    #get new features
    new_merchant_transactions = pd.read_csv('./data/raw/new_merchant_transactions.csv', parse_dates=['purchase_date'])
    new_merch_agg = get_merch_features(new_merchant_transactions[['card_id', 'merchant_id']])
    new_merchant_transactions = get_transaction_features(new_merchant_transactions)
    new_merchant_transactions = pd.merge(new_merchant_transactions, new_merch_agg, on='card_id', how='left')
    new_merchant_transactions.columns = ['new_' + c if c != 'card_id' else c for c in new_merchant_transactions.columns]
    new_merchant_transactions.to_csv('./data/processed/new_merchant_transactions_agg_v8.csv')

    train=pd.read_csv('./data/raw/train.csv')
    train = pd.merge(train, historical_transactions, on='card_id', how='left')
    train = pd.merge(train, new_merchant_transactions, on='card_id', how='left')
    train = get_final_features(train)
    train.to_csv('./data/processed/train_transactions_agg_v8.csv')

    test=pd.read_csv('./data/raw/test.csv')
    test = pd.merge(test, historical_transactions, on='card_id', how='left')
    test = pd.merge(test, new_merchant_transactions, on='card_id', how='left')
    test = get_final_features(test)
    test.to_csv('./data/processed/test_transactions_agg_v8.csv')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()