import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import lightgbm as lgb
from math import sqrt
import csv
from random import randint
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import pickle


def final_submission():
	df=list(range(0,40))
	df[1] = pd.read_csv('./reports/submission_0_20170101.csv',dtype={'fullVisitorId': 'str'})  
	df[2] = pd.read_csv('./reports/submission_0_20170701.csv',dtype={'fullVisitorId': 'str'})  
	df[3] = pd.read_csv('./reports/submission_1_20161201.csv',dtype={'fullVisitorId': 'str'})  
	df[4] = pd.read_csv('./reports/submission_1_20170601.csv',dtype={'fullVisitorId': 'str'})  
	df[5] = pd.read_csv('./reports/submission_2_20161101.csv',dtype={'fullVisitorId': 'str'})  
	df[6] = pd.read_csv('./reports/submission_2_20170501.csv',dtype={'fullVisitorId': 'str'})
	df[7] = pd.read_csv('./reports/submission_0_20160801.csv',dtype={'fullVisitorId': 'str'})  
	df[8] = pd.read_csv('./reports/submission_0_20170201.csv',dtype={'fullVisitorId': 'str'})  
	df[9] = pd.read_csv('./reports/submission_0_20170801.csv',dtype={'fullVisitorId': 'str'})  
	df[10] = pd.read_csv('./reports/submission_1_20170101.csv',dtype={'fullVisitorId': 'str'})  
	df[11] = pd.read_csv('./reports/submission_1_20170701.csv',dtype={'fullVisitorId': 'str'})  
	df[12] = pd.read_csv('./reports/submission_2_20161201.csv',dtype={'fullVisitorId': 'str'})  
	df[13] = pd.read_csv('./reports/submission_2_20170601.csv',dtype={'fullVisitorId': 'str'})
	df[14] = pd.read_csv('./reports/submission_0_20160901.csv',dtype={'fullVisitorId': 'str'})  
	df[15] = pd.read_csv('./reports/submission_0_20170301.csv',dtype={'fullVisitorId': 'str'})  
	df[16] = pd.read_csv('./reports/submission_1_20160801.csv',dtype={'fullVisitorId': 'str'})  
	df[17] = pd.read_csv('./reports/submission_1_20170201.csv',dtype={'fullVisitorId': 'str'})  
	df[18] = pd.read_csv('./reports/submission_1_20170801.csv',dtype={'fullVisitorId': 'str'})  
	df[19] = pd.read_csv('./reports/submission_2_20170101.csv',dtype={'fullVisitorId': 'str'})  
	df[20] = pd.read_csv('./reports/submission_2_20170701.csv',dtype={'fullVisitorId': 'str'})
	df[21] = pd.read_csv('./reports/submission_0_20161001.csv',dtype={'fullVisitorId': 'str'})  
	df[22] = pd.read_csv('./reports/submission_0_20170401.csv',dtype={'fullVisitorId': 'str'})  
	df[23] = pd.read_csv('./reports/submission_1_20160901.csv',dtype={'fullVisitorId': 'str'})  
	df[24] = pd.read_csv('./reports/submission_1_20170301.csv',dtype={'fullVisitorId': 'str'})  
	df[25] = pd.read_csv('./reports/submission_2_20160801.csv',dtype={'fullVisitorId': 'str'})  
	df[26] = pd.read_csv('./reports/submission_2_20170201.csv',dtype={'fullVisitorId': 'str'})  
	df[27] = pd.read_csv('./reports/submission_2_20170801.csv',dtype={'fullVisitorId': 'str'})
	df[28] = pd.read_csv('./reports/submission_0_20161101.csv',dtype={'fullVisitorId': 'str'})  
	df[29] = pd.read_csv('./reports/submission_0_20170501.csv',dtype={'fullVisitorId': 'str'})  
	df[30] = pd.read_csv('./reports/submission_1_20161001.csv',dtype={'fullVisitorId': 'str'})  
	df[31] = pd.read_csv('./reports/submission_1_20170401.csv',dtype={'fullVisitorId': 'str'})  
	df[32] = pd.read_csv('./reports/submission_2_20160901.csv',dtype={'fullVisitorId': 'str'})  
	df[33] = pd.read_csv('./reports/submission_2_20170301.csv',dtype={'fullVisitorId': 'str'})
	df[34] = pd.read_csv('./reports/submission_0_20161201.csv',dtype={'fullVisitorId': 'str'})  
	df[35] = pd.read_csv('./reports/submission_0_20170601.csv',dtype={'fullVisitorId': 'str'})  
	df[36] = pd.read_csv('./reports/submission_1_20161101.csv',dtype={'fullVisitorId': 'str'})  
	df[37] = pd.read_csv('./reports/submission_1_20170501.csv',dtype={'fullVisitorId': 'str'})  
	df[38] = pd.read_csv('./reports/submission_2_20161001.csv',dtype={'fullVisitorId': 'str'})  
	df[39] = pd.read_csv('./reports/submission_2_20170401.csv',dtype={'fullVisitorId': 'str'})

	print(df[1].shape)
	final_submission = df[1]
	for i in range(2, 40):
		final_submission = pd.merge(final_submission, df[i], on='fullVisitorId', suffixes=('_'+str(i-1), '_'+str(i)))

	print(final_submission.info())
	print(final_submission.shape)
	final_submission_df = final_submission.set_index('fullVisitorId').mean(axis=1).reset_index()
	final_submission_df.columns=['fullVisitorId', 'PredictedLogRevenue']
	print(final_submission_df.shape)
	print(final_submission_df.info())
	print(final_submission_df.describe())
	final_submission_df.to_csv('./reports/all_submission.csv', index=False)

	df_1 = pd.read_csv('./reports/submission_0_20170501.csv',dtype={'fullVisitorId': 'str'})
	df_1.columns=['fullVisitorId', 'PredictedLogRevenue_0']
	df_2 = pd.read_csv('./reports/submission_1_20170501.csv',dtype={'fullVisitorId': 'str'})
	df_2.columns=['fullVisitorId', 'PredictedLogRevenue_1']
	df_3 = pd.read_csv('./reports/submission_2_20170501.csv',dtype={'fullVisitorId': 'str'})
	df_3.columns=['fullVisitorId', 'PredictedLogRevenue_2']

	only_may_submission = pd.merge(df_1, df_2, on='fullVisitorId')
	only_may_submission = pd.merge(only_may_submission, df_3, on='fullVisitorId')
	only_may_submission = only_may_submission.set_index('fullVisitorId').mean(axis=1).reset_index()
	only_may_submission.columns=['fullVisitorId', 'PredictedLogRevenue']

	print(only_may_submission.info())
	print(only_may_submission.describe())
	print(only_may_submission[only_may_submission.PredictedLogRevenue>0].shape[0]/only_may_submission.shape[0])
	
	only_may_submission.to_csv('./reports/only_may_submission.csv', index=False)

################################
# main function 
################################
@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features and target variable')
    final_submission()



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()