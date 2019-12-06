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

def add_diff_features(df):
    print("start")
    df = df.fillna(0)

    df['diff_purchase_sum_13_12'] = df['hist_purchase_amount_sum_-12'] - df['hist_purchase_amount_sum_-13']
    df['diff_purchase_sum_12_11'] = df['hist_purchase_amount_sum_-11'] - df['hist_purchase_amount_sum_-12']
    df['diff_purchase_sum_11_10'] = df['hist_purchase_amount_sum_-10'] - df['hist_purchase_amount_sum_-11']
    df['diff_purchase_sum_10_9'] = df['hist_purchase_amount_sum_-9'] - df['hist_purchase_amount_sum_-10']
    df['diff_purchase_sum_9_8'] = df['hist_purchase_amount_sum_-8'] - df['hist_purchase_amount_sum_-9']
    df['diff_purchase_sum_8_7'] = df['hist_purchase_amount_sum_-7'] - df['hist_purchase_amount_sum_-8']
    df['diff_purchase_sum_7_6'] = df['hist_purchase_amount_sum_-6'] - df['hist_purchase_amount_sum_-7']
    df['diff_purchase_sum_6_5'] = df['hist_purchase_amount_sum_-5'] - df['hist_purchase_amount_sum_-6']
    df['diff_purchase_sum_5_4'] = df['hist_purchase_amount_sum_-4'] - df['hist_purchase_amount_sum_-5']
    df['diff_purchase_sum_4_3'] = df['hist_purchase_amount_sum_-3'] - df['hist_purchase_amount_sum_-4']
    df['diff_purchase_sum_3_2'] = df['hist_purchase_amount_sum_-2'] - df['hist_purchase_amount_sum_-3']
    df['diff_purchase_sum_2_1'] = df['hist_purchase_amount_sum_-1'] - df['hist_purchase_amount_sum_-2']
    df['diff_purchase_sum_1_0'] = df['hist_purchase_amount_sum_0'] - df['hist_purchase_amount_sum_-1']
    df = df.fillna(0)
    print("mean")
    df['diff_purchase_lag_mean'] = df[['diff_purchase_sum_13_12', 'diff_purchase_sum_12_11','diff_purchase_sum_11_10', 'diff_purchase_sum_10_9', 'diff_purchase_sum_9_8', 'diff_purchase_sum_8_7', 'diff_purchase_sum_7_6', 'diff_purchase_sum_6_5', 'diff_purchase_sum_5_4', 'diff_purchase_sum_4_3', 'diff_purchase_sum_3_2','diff_purchase_sum_2_1', 'diff_purchase_sum_1_0']].mean(axis=1)
    
    df['diff_merchant_id_nunique_5_4'] = df['hist_merchant_id_nunique_-4'] - df['hist_merchant_id_nunique_-5']
    df['diff_merchant_id_nunique_4_3'] = df['hist_merchant_id_nunique_-3'] - df['hist_merchant_id_nunique_-4']
    df['diff_merchant_id_nunique_3_2'] = df['hist_merchant_id_nunique_-2'] - df['hist_merchant_id_nunique_-3']
    df['diff_merchant_id_nunique_2_1'] = df['hist_merchant_id_nunique_-1'] - df['hist_merchant_id_nunique_-2']
    df['diff_merchant_id_nunique_1_0'] = df['hist_merchant_id_nunique_0'] - df['hist_merchant_id_nunique_-1']
    df = df.fillna(0)
    print("mean")
    df['diff_merchant_id_lag_mean'] = df[['diff_merchant_id_nunique_5_4', 'diff_merchant_id_nunique_4_3', 'diff_merchant_id_nunique_3_2', 'diff_merchant_id_nunique_2_1', 'diff_merchant_id_nunique_1_0']].mean(axis=1)

    df['diff_subsector_id_nunique_5_4'] = df['hist_subsector_id_nunique_-4'] - df['hist_subsector_id_nunique_-5']
    df['diff_subsector_id_nunique_4_3'] = df['hist_subsector_id_nunique_-3'] - df['hist_subsector_id_nunique_-4']
    df['diff_subsector_id_nunique_3_2'] = df['hist_subsector_id_nunique_-2'] - df['hist_subsector_id_nunique_-3']
    df['diff_subsector_id_nunique_2_1'] = df['hist_subsector_id_nunique_-1'] - df['hist_subsector_id_nunique_-2']
    df['diff_subsector_id_nunique_1_0'] = df['hist_subsector_id_nunique_0'] - df['hist_subsector_id_nunique_-1']
    df = df.fillna(0)
    print("mean")
    df['diff_subsector_id_lag_mean'] = df[['diff_subsector_id_nunique_5_4','diff_subsector_id_nunique_4_3','diff_subsector_id_nunique_3_2','diff_subsector_id_nunique_2_1','diff_subsector_id_nunique_1_0']].mean(axis=1)
    df['diff_purchase_by_merchant'] = df['diff_purchase_lag_mean']/(df['diff_merchant_id_lag_mean']+1)
    df['diff_purchase_by_subsector'] = df['diff_subsector_id_lag_mean']/(df['diff_merchant_id_lag_mean']+1)

    print("done")

    return df


def add_merchant_lag_features():
    print('merch_purchase_diff')
    df = pd.read_csv('./data/raw/historical_transactions.csv', parse_dates=['purchase_date'])
    aggs = {
        'purchase_amount': ['sum']                            
    }
    transactions_by_lag = df.groupby(['card_id','merchant_id', 'month_lag']).agg(aggs)
    transactions_by_lag.columns = ['_'.join(col).strip() for col in transactions_by_lag.columns.values]
    transactions_by_lag = transactions_by_lag.reset_index()
    transactions_by_lag = transactions_by_lag.fillna(0)
    transactions_by_lag = transactions_by_lag.set_index(['card_id','merchant_id'])
    transactions_by_lag = transactions_by_lag.pivot(columns='month_lag')
    transactions_by_lag.columns =[(col[0]+'_'+str(col[1])).strip() for col in transactions_by_lag.columns.values]
    transactions_by_lag = transactions_by_lag.reset_index()
    transactions_by_lag = transactions_by_lag.fillna(0)
    transactions_by_lag['merch_purchase_diff_0_1'] = transactions_by_lag['purchase_amount_sum_0'] - transactions_by_lag['purchase_amount_sum_-1'] 
    transactions_by_lag['merch_purchase_diff_1_2'] = transactions_by_lag['purchase_amount_sum_-1'] - transactions_by_lag['purchase_amount_sum_-2'] 
    transactions_by_lag['merch_purchase_diff_2_3'] = transactions_by_lag['purchase_amount_sum_-2'] - transactions_by_lag['purchase_amount_sum_-3'] 
    transactions_by_lag['merch_purchase_diff_3_4'] = transactions_by_lag['purchase_amount_sum_-3'] - transactions_by_lag['purchase_amount_sum_-4'] 
    transactions_by_lag['merch_purchase_diff_4_5'] = transactions_by_lag['purchase_amount_sum_-4'] - transactions_by_lag['purchase_amount_sum_-5'] 
    transactions_by_lag['merch_purchase_diff_5_6'] = transactions_by_lag['purchase_amount_sum_-5'] - transactions_by_lag['purchase_amount_sum_-6'] 
    transactions_by_lag['merch_purchase_diff_lag_mean'] = transactions_by_lag[['merch_purchase_diff_0_1','merch_purchase_diff_1_2', 'merch_purchase_diff_2_3', 'merch_purchase_diff_3_4', 'merch_purchase_diff_4_5', 'merch_purchase_diff_5_6']].mean(axis=1)
    transactions_by_lag = transactions_by_lag.groupby('card_id').mean().reset_index()

    '''
    print('lag_diff')
    df =  df[['card_id', 'merchant_id', 'month_lag']].drop_duplicates()
    print("sort_values")
    df = df.sort_values(by=['card_id','merchant_id', 'month_lag'])
    print("sorting done")
    df['lag_diff'] = df.groupby(['card_id', 'merchant_id']).diff()
    df = df.fillna(0)
    df = df.groupby('card_id')['lag_diff'].mean().reset_index()
    '''
    print('lag_diff')
    df =  df[['card_id', 'merchant_id', 'month_lag']].drop_duplicates()
    df = df.groupby(['card_id', 'merchant_id'])['month_lag'].mean().reset_index()
    df = df.groupby('card_id')['month_lag'].mean().reset_index()
    transactions_by_lag = pd.merge(transactions_by_lag, df, on='card_id', how='left')

    return transactions_by_lag

@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    merch_diff = add_merchant_lag_features()
    # get hist features
    train = pd.read_csv('./data/processed/train_transactions_agg_v8.csv')
    train = add_diff_features(train)
    train = pd.merge(train, merch_diff, on='card_id', how='left')
    train.to_csv('./data/processed/train_transactions_agg_v9.csv')

    test = pd.read_csv('./data/processed/test_transactions_agg_v8.csv')
    test = add_diff_features(test)
    test = pd.merge(test, merch_diff, on='card_id', how='left')
    test.to_csv('./data/processed/test_transactions_agg_v9.csv')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()