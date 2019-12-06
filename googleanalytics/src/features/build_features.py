import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import time
from datetime import date, timedelta

def remove_const_columns(df):
	cols = ['socialEngagementType', 'device_browserSize', 'device_browserVersion',
       'device_flashVersion', 'device_language', 'device_mobileDeviceBranding',
       'device_mobileDeviceInfo', 'device_mobileDeviceMarketingName',
       'device_mobileDeviceModel', 'device_mobileInputSelector',
       'device_operatingSystemVersion', 'device_screenColors',
       'device_screenResolution', 'geoNetwork_cityId', 'geoNetwork_latitude',
       'geoNetwork_longitude', 'geoNetwork_networkLocation', 'totals_visits',
       'trafficSource_adwordsClickInfo.criteriaParameters'
       ]
	df = df.drop(cols, axis=1)
	return df

def fillnull(df):
    # fill na
    df['totals_bounces'] = df['totals_bounces'].fillna(0)
    df['totals_newVisits'] = df['totals_newVisits'].fillna(0)
    df['totals_pageviews'] = df['totals_pageviews'].fillna(1)
    df.loc[((df['totals_sessionQualityDim'].isnull()) & (df['totals_transactions'] > 0)), 'totals_sessionQualityDim'] = 100
    df.loc[((df['totals_sessionQualityDim'].isnull()) & (df['totals_totalTransactionRevenue'] > 0)), 'totals_sessionQualityDim'] = 100
    df.loc[((df['totals_sessionQualityDim'].isnull()) & (df['totals_transactionRevenue'] > 0)), 'totals_sessionQualityDim'] = 100
    df.loc[((df['totals_sessionQualityDim'].isnull()) & (df['totals_timeOnSite'] > 4000)), 'totals_sessionQualityDim'] = 70
    df.loc[((df['totals_sessionQualityDim'].isnull()) & (df['totals_timeOnSite'] > 2000)), 'totals_sessionQualityDim'] = 60
    df['totals_sessionQualityDim'] = df['totals_sessionQualityDim'].fillna(1)

    df.loc[((df['totals_timeOnSite'].isnull()) & (df['totals_transactions'] > 0)), 'totals_timeOnSite'] = 794.0
    df.loc[((df['totals_timeOnSite'].isnull()) & (df['totals_totalTransactionRevenue'] > 0)), 'totals_timeOnSite'] = 794.0
    df.loc[((df['totals_timeOnSite'].isnull()) & (df['totals_transactionRevenue'] > 0)), 'totals_timeOnSite'] = 794.0
    df['totals_timeOnSite'] = df['totals_timeOnSite'].fillna(93.0)

    df['totals_totalTransactionRevenue'] = df['totals_totalTransactionRevenue'].fillna(0)
    df['totals_transactionRevenue'] = df['totals_transactionRevenue'].fillna(0)
    
    df.loc[((df['totals_transactions'].isnull()) & (df['totals_totalTransactionRevenue'] > 0)), 'totals_transactions'] = 1
    df.loc[((df['totals_transactions'].isnull()) & (df['totals_transactionRevenue'] > 0)), 'totals_transactions'] = 1
    df['totals_transactions'] = df['totals_transactions'].fillna(0)

    return df

def handle_categorical(df):

	df['is_dektop'] = 0
	df.loc[df.device_deviceCategory.isin(['desktop', 'Desktop']), 'is_dektop'] = 1
	
	valid_browsers = ['Chrome','Firefox','Internet Explorer','Safari','Edge','Opera','Safari (in-app)','Samsung Internet','Android Webview']                               
	df['valid_browser'] = 0
	df.loc[df.device_browser.isin(valid_browsers), 'valid_browser'] = 1
	df['is_chrome'] = 0
	df.loc[df.device_browser.isin(['Chrome', 'chrome']), 'is_chrome'] = 1
	
	df['is_america'] = 0
	df.loc[df.geoNetwork_continent.isin(['Americas', 'americas']), 'is_america'] = 1

	df['is_country_st_lucia'] = 0
	df.loc[df.geoNetwork_country.isin(['St. Lucia', 'st. Lucia', 'st lucia']), 'is_country_st_lucia'] = 1
	
	df['is_country_us'] = 0
	df.loc[df.geoNetwork_country.isin(['United States', 'united states', 'us']), 'is_country_us'] = 1
	
	city_1 = ['Richmond', 'Las Vegas', 'Edison', 'Tacoma', 'Walnut Creek','Chelmsford', 'Dundalk', 'Council Bluffs']
	city_2 = ['Reston', 'Ann Arbor', 'Lake Oswego', 'Milwaukee', 'San Bruno', 'Bellevue', 'Baltimore', 'Kirkland', 'Cambridge', 'Austin']
	city_3 = ['New York', 'Berkeley', 'Seattle', 'Chicago', 'Milpitas', 'Irvine','Sunnyvale', 'Boulder', 'Barrie', 'San Marcos', 'Mountain View', 'Cupertino', 'Oakland', 'Atlanta', 'Jersey City', 'San Francisco']
	city_4 = ['Oklahoma City', 'Salem', 'Los Angeles', 'Calgary', 'Kansas City','Fremont', 'Pittsburgh', 'Santa Clara', 'Palo Alto','Redwood City', 'San Mateo', 'Birmingham', 'Portland', 'San Diego','Fresno', 'San Jose', 'Washington', 'Minneapolis', 'Chico',
    'Omaha', 'Santa Monica', 'South San Francisco', 'Denver','Nashville', 'Boston', 'Dallas', 'Houston', 'Columbus', 'Raleigh']

	df['in_city_1'] = 0
	df.loc[df.geoNetwork_city.isin(city_1), 'in_city_1'] = 1
	df['in_city_2'] = 0
	df.loc[df.geoNetwork_city.isin(city_2), 'in_city_2'] = 1
	df['in_city_3'] = 0
	df.loc[df.geoNetwork_city.isin(city_3), 'in_city_3'] = 1
	df['in_city_4'] = 0
	df.loc[df.geoNetwork_city.isin(city_4), 'in_city_4'] = 1   

	return df

def get_date_features(df):
	    # Add data features
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_dom'] = df['date'].dt.day
    #df['sess_date_hour'] = df['date'].dt.hour
    print('dates columns added')

    return df

#['is_dektop','valid_browser', 'is_chrome', 'is_america',  'is_country_st_lucia', 'is_country_us', 'in_city_1', 'in_city_2', 'in_city_3', 'in_city_4']
def aggregate_sessions(df, end_date, cat_feats=[], sum_of_logs=False):
    """
    Aggregate session data for each fullVisitorId
    :param df: DataFrame to aggregate on
    :param cat_feats: List of Categorical features
    :param sum_of_logs: if set to True, revenues are first log transformed and then summed up  
    :return: aggregated fullVisitorId data over Sessions
    """

    print('start aggregate_sessions')
    if sum_of_logs is True:
        # Go to log first
        df['transactionRevenue'] = np.log1p(df['totals_transactionRevenue'])

    aggs = {
    	'date': ['min', 'max'],
		'visitNumber': ['max', 'size'],
		'totals_bounces': ['sum', 'mean', 'median'],                                        
		'totals_hits': ['sum', 'min', 'max', 'mean', 'median'],                                           
		'totals_newVisits': ['sum', 'mean', 'median'],                                   
		'totals_pageviews': ['sum', 'min', 'max', 'mean', 'median'],                                     
		'totals_sessionQualityDim': ['sum', 'min', 'max', 'mean', 'median'],                               
		'totals_timeOnSite': ['sum', 'min', 'max', 'mean', 'median'],                                      
		'totals_totalTransactionRevenue':  ['sum', 'min', 'max', 'mean', 'median'],                     
		'totals_transactionRevenue': ['sum', 'min', 'max', 'mean', 'median'],                            
		'totals_transactions':  ['sum', 'min', 'max', 'mean', 'median'],                                  
    }

    for f in ['sess_date_dow', 'sess_date_dom']:
        aggs[f] = ['min', 'max', 'mean', 'median', 'var', 'std', 'sum']

    for f in cat_feats: 
        aggs[f] = ['mean', 'sum']

    users = df.groupby('fullVisitorId').agg(aggs)
    print('User aggregation done')

    # This may not work in python 3.5, since keys ordered is not guaranteed
    new_columns = ['_'.join(col).strip() for col in users.columns]

    print('New columns are : {}'.format(new_columns))
    users.columns = new_columns

    # Add dates
    end_date = pd.to_datetime(end_date, format='%Y%m%d')
    users['date_diff'] = (users.date_max - users.date_min).astype(np.int64)
    users['days_since_last_visit'] = (end_date - users.date_max).astype(np.int64)
    users['days_since_first_visit']= (end_date - users.date_min).astype(np.int64)
    users['visit_frequency'] = users['visitNumber_max']/users['date_diff']

    # last 45 days and 90 days features
    date_45 =  end_date-timedelta(45)
    date_90 =  end_date-timedelta(90)
    df_45 = df[df.date >= date_45]
    df_90 = df[df.date >= date_90]

    aggs_recent = {
		'totals_hits': ['sum'],                                           
		'totals_pageviews': ['sum'],                                     
		'totals_sessionQualityDim': ['sum'],                               
		'totals_timeOnSite': ['sum']                                                                       
    }

    # agg 45 days
    users_45 = df_45.groupby('fullVisitorId').agg(aggs_recent)
    print('User aggregation done')

    # This may not work in python 3.5, since keys ordered is not guaranteed
    new_columns = ['_45_'.join(col).strip() for col in users_45.columns]

    print('New columns are : {}'.format(new_columns))
    users_45.columns = new_columns

    # agg 90 days
    users_90 = df_90.groupby('fullVisitorId').agg(aggs_recent)
    print('User aggregation done')

    # This may not work in python 3.5, since keys ordered is not guaranteed
    new_columns = ['_90_'.join(col).strip() for col in users_90.columns]

    print('New columns are : {}'.format(new_columns))
    users_90.columns = new_columns

    users = users.reset_index()
    users = pd.merge(users, users_45, on='fullVisitorId', how='left')
    users = pd.merge(users, users_90, on='fullVisitorId', how='left')

    # Go to log if not already done
    if sum_of_logs is False:
        # Go to log first
        users['totals_transactionRevenue_sum'] = np.log1p(users['totals_transactionRevenue_sum'])

    return users
                                 
def get_target_transaction_variable(df, start_date, end_date):
	df = df[(df.date >=start_date) & (df.date <= end_date)].groupby('fullVisitorId')['totals_transactionRevenue'].sum().reset_index()
	df.columns = ['fullVisitorId', 'target']
	df['target'] = np.log1p(df['target'])
	
	return df

def get_target_comeback_prediction(df, start_date, end_date):
	df = df[(df.date >=start_date) & (df.date <= end_date)].groupby('fullVisitorId').size().reset_index() 
	df.columns = ['fullVisitorId', 'target']

	return df

def get_x_data(df, start_date, end_date):
	df = df[(df.date >=start_date) & (df.date <= end_date)]
	df = get_date_features(df)
	df = aggregate_sessions(df, end_date)

	return df

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('target')
@click.argument('start_date')
def main(input_filepath, output_filepath, target, start_date):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features and target variable')

    start_date = int(start_date)
    
    if(target == 'None'):

    	# get start and end date for x variables
    	t=time.strptime(str(start_date),'%Y%m%d')
    	end_date=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(167)
    	end_date = int(end_date.strftime('%Y%m%d'))
    	print(start_date, end_date)

    	df = pd.read_csv(input_filepath, dtype={'fullVisitorId': 'str'})
    	logger.info('loaded input file')
    	df = remove_const_columns(df)
    	df = handle_categorical(df)
    	df = fillnull(df)
    	df = get_x_data(df, start_date, end_date)
    	print(df.columns)
    	df.to_csv(output_filepath, index=False)

    if(target == 'totals_transactions'):

    	# get the start and end date for predictions
    	t=time.strptime(str(start_date),'%Y%m%d')
    	start_date = date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(167+47)
    	start_date = int(start_date.strftime('%Y%m%d'))
    	end_date=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(167+47+61)
    	end_date = int(end_date.strftime('%Y%m%d'))
    	print(start_date, end_date)

    	df = pd.read_csv(input_filepath, dtype={'fullVisitorId': 'str'})
    	logger.info('loaded input file')
    	df = remove_const_columns(df)
    	df = handle_categorical(df)
    	df = fillnull(df)    	
    	df = get_target_transaction_variable(df, start_date, end_date)    	
    	df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

