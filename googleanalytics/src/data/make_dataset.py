# -*- coding: utf-8 -*-
import click
import logging
import json
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import os
from ast import literal_eval

def load_df_wo_array(input_filepath, output_filepath, nrows=None):
    USE_COLUMNS = [
        'channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
        'socialEngagementType', 'totals', 'trafficSource', 'visitId',
        'visitNumber', 'visitStartTime'
    ]

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(input_filepath,
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, nrows=nrows, usecols=USE_COLUMNS)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    print(f"Loaded {os.path.basename(input_filepath)}. Shape: {df.shape}")
    df.to_csv(output_filepath, index=False)

def load_df_array(input_filepath, output_filepath, key_columns, array_column, chunk=100000):
    USE_COLUMNS = key_columns+[array_column]
    reader = pd.read_csv(input_filepath, 
                     dtype={'fullVisitorId': 'str'}, 
                     usecols=USE_COLUMNS, chunksize = chunk, skiprows=0)
    i = rows = 0
    for df in reader:
        df[array_column][df[array_column] == "[]"] = "[{}]"
        df[array_column]=df[array_column].apply(literal_eval)
        df[key_columns] = df[key_columns].astype(str)
        df['key'] = df[key_columns].apply(lambda x: '_'.join(x), axis=1)
        df = df.drop(key_columns, axis=1)
        df = df.join(df[array_column].apply(pd.Series)).drop(array_column, 1).set_index([u'key']).stack().reset_index().drop('level_1', 1).rename(columns={0:array_column})
        column_as_df = json_normalize(df[array_column])
        column_as_df.columns = [f"{array_column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(array_column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
        df.to_csv(f"{output_filepath}_{i:03d}.csv", index=False)
        print(f"Loaded {os.path.basename(input_filepath)}. Shape: {df.shape}, columns: {USE_COLUMNS}-{i:03d}, rows: {rows}")
        rows += len(df.index)
        i = i+1
    
#df = load_df_array('train_v2.csv', 'a_out.csv', ['fullVisitorId', 'visitId'], 'hits', nrows=2)
#df = load_df_array('train_v2.csv', 'a_out.csv', ['fullVisitorId', 'visitId'], 'customDimensions', nrows=2)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('columns')
@click.argument('nrows')
def main(input_filepath, output_filepath, columns, nrows):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    if(columns == 'None'):
        load_df_wo_array(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
