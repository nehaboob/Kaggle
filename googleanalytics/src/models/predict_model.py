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


##################################################
# preprocessing and training
##################################################
def preprocess_x(x_df):
	#remove constant columns 
	const_cols = [c for c in x_df.columns if x_df[c].nunique(dropna=False)==1 ]
	cols = np.setdiff1d(x_df.columns.values, const_cols)
	x_df = x_df[cols]

	# remove object columns
	const_cols = [c for c in x_df.columns if x_df[c].dtype.name == 'object']
	cols = np.setdiff1d(x_df.columns.values, const_cols)
	cols = np.append(cols, ['fullVisitorId'])
	x_df = x_df[cols]

	# remove less important columns
	"""
	less_imp = ['sess_date_dom_max', 'sess_date_dom_mean', 'sess_date_dom_median',
 	'sess_date_dom_min', 'sess_date_dow_max', 'sess_date_dow_min', 'sess_date_dow_median',
 	'sess_date_dow_var', 'totals_bounces_median', 'totals_bounces_sum',
 	'totals_hits_min', 'totals_newVisits_median', 'totals_newVisits_sum',
 	'totals_pageviews_min', 'totals_transactions_max',
 	'totals_transactions_median', 'totals_transactions_min',
 	'totals_transactions_sum']
	
	cols = np.setdiff1d(x_df.columns.values, less_imp)
	x_df = x_df[cols]
	"""
	return x_df

def generate_predictions(x, model):
	fullVisitorId = x['fullVisitorId'].values
	cols = list(x.columns.values)
	cols.remove('fullVisitorId')
	print(cols)
	X = x[cols]
	th = 95
	y_pred = model.predict(X)
	thresh = np.percentile(y_pred, th)
	y_pred[y_pred < thresh] = 0

	predict_df = pd.DataFrame({'fullVisitorId':fullVisitorId , 'PredictedLogRevenue': y_pred})

	return predict_df

################################
# main function 
################################
@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, model_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features and target variable')
    x = pd.read_csv(input_filepath, dtype={'fullVisitorId': 'str'})
    x = preprocess_x(x)

    model = pickle.load(open(model_filepath, 'rb'))
    prediction_df = generate_predictions(x, model)
    prediction_df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()