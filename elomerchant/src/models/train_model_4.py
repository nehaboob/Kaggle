import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# reg_lambda higher for overfitting
# reg_alpha higher for overfitting
global ITERATION

def get_my_config():	
	exp_desc = 'WIth SKF files and lag level features'
	algo = 'LGBM'
	return exp_desc, algo

###################################
# logging and helper function
###################################
def cal_rmse(y_true, y_predict):
	rms = sqrt(mean_squared_error(y_true, y_predict))
	return rms

def get_scores(y_true, y_predict, mode):
    scores_df = pd.Series()
    scores_df[mode+'_rmse'] = cal_rmse(y_true, y_predict)
    return scores_df

def log_metrics(scores_df, train_scores, feature_importances, lgb_params, fold, exp_id):
    n = randint(0, 10000)
    exp_desc, algo = get_my_config()

    with open('results_metrics.csv', 'a') as f:
    	spamwriter = csv.writer(f)
    	spamwriter.writerow([
                            exp_id,
                            n,
                            exp_desc,
                            algo,
                            lgb_params,
                            fold,
                            train_scores['train_rmse'],
                            scores_df['test_rmse']
						])

    with open('results_params_imps.csv', 'a') as f:
    	spamwriter = csv.writer(f)
    	spamwriter.writerow([
                            exp_id,
                            n,
                            exp_desc,
                            algo,
							lgb_params, 
							fold,
                            feature_importances
						])




##################################################
# preprocessing and training
##################################################


# XGB training 
# take data and params and return the loss to minimize
# log the metrics

def train_XGB(X, y, params, splits, store_results=False, exp_id=None, test_results=False, X_test=None):
	start = timer()

	exp_desc, algo = get_my_config()
	oof_reg_preds = np.zeros(X.shape[0])
	if(test_results):
		test_pred = np.zeros(X_test.shape[0])
	print(params)
	feature_imp=pd.Series(index= X.columns.values, data=0)

	# other scikit-learn modules
	estimator = lgb.LGBMRegressor(boosting='gbdt', objective='regression')

	param_grid = {
			   'n_estimators': [8000, 10000, 15000],
               'max_depth':  [4, 8, -1],
               'num_leaves': [31],
               'subsample': [0.6, 0.8, 1.0],
               'colsample_bytree': [0.6],
               'early_stopping_rounds':['100']
               }

	gbm = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=10)
	gbm.fit(X, y)

	print('Best parameters found by grid search are:', gbm.best_params_)
	print("BEST CV SCORE: " + str(gbm.best_score_))

	return True                  


################################
# main function 
################################
@click.command()
@click.argument('type')
def main(type):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features and target variable')
    train=pd.read_csv('./data/processed/train_transactions_agg_v9.csv')
    train=train.fillna(0)

    test=pd.read_csv('./data/processed/test_transactions_agg_v9.csv')
    test=test.fillna(0)

    cols = [col for col in train.columns if col not in ['first_active_month', 'card_id', 'target', 'Unnamed: 0']]
    cols = ['new_category_1_sum','new_category_1_mean','new_subsect_purchase_mean_ratio','dayofweek','new_purchase_mean_month_diff','new_merch_purchase_mean_ratio','hist_merchant_id_nunique_-9','hist_merch_month_diff','hist_weekend_mean','purchase_amount_sum_-7','hist_purchase_amount_max_-2','feature_1','new_category_2_5.0_mean','diff_purchase_sum_6_5','hist_purchase_amount_min_0','new_subsector_id_nunique_1','new_purchase_amount_max_2','hist_first_buy','hist_merchant_id_count_-9','merch_purchase_diff_5_6','hist_purchase_amount_max','hist_purchase_amount_max_-1','new_merch_avg_sales_lag3_std','new_purchase_amount_min_1','hist_purchase_date_uptonow','diff_purchase_lag_mean','hist_installments_sum_0','hist_subsector_id_nunique_0','diff_purchase_sum_7_6','purchase_amount_sum_0','new_merch_numerical_2_min','new_merchant_category_id_nunique','hist_authorized_flag_mean','new_purchase_date_min','hist_purchase_amount_min_-4','hist_category_3_B_sum','new_purchase_month_mean','merch_purchase_diff_lag_mean','merch_purchase_diff_4_5','new_month_diff_mean','hist_purchase_amount_std_-4','hist_month_lag_nunique','hist_purchase_amount_mean_-3','new_purchase_date_ptp_ptp','hist_purchase_amount_mean_0','hist_merch_avg_purchases_lag3_max','hist_purchase_amount_sum_-5','hist_merchant_id_nunique_-3','new_purchase_amount_std','purchase_amount_total','purchase_amount_sum_-3','diff_subsector_id_nunique_5_4','hist_merchant_category_id_nunique_-7','purchase_amount_sum_-5','hist_subsector_id_nunique_-9','hist_purchase_amount_std_-3','hist_merchant_category_id_nunique_-5','diff_purchase_sum_9_8','hist_purchase_amount_min','hist_purchase_month_std','hist_category_1_sum','new_purchase_amount_max','hist_purchase_amount_max_0','hist_merchant_id_nunique_-10','hist_purchase_amount_sum_-4','diff_purchase_sum_8_7','hist_purchase_amount_max_-3','hist_installments_mean_0','hist_purchase_amount_mean_-6','hist_installments_mean_-11','hist_category_2_1.0_mean','hist_installments_mean_-4','diff_merchant_id_nunique_1_0','diff_purchase_sum_10_9','new_first_buy','hist_purchase_date_average','hist_merch_avg_purchases_lag12_max','new_merch_purchase_sum_ratio','hist_purchase_amount_mean_-5','new_city_id_nunique','hist_purchase_amount_min_-5','hist_merch_purchase_sum_ratio','hist_merchant_id_nunique_-4','hist_merchant_category_id_nunique_-9','hist_merch_purchase_mean_ratio','hist_purchase_date_min','hist_purchase_amount_min_-3','hist_merch_avg_sales_lag3_mean','hist_installments_sum','hist_installments_mean_-3','hist_purchase_amount_sum_-1','new_merch_avg_sales_lag6_sum','new_purchase_date_uptonow','hist_installments_mean_-8','hist_installments_std_-6','hist_purchase_amount_sum_-10','hist_purchase_amount_sum_0','elapsed_time','new_category_3_A_sum','new_purchase_date_average','hist_purchase_amount_max_-4','hist_category_3_C_mean','hist_subsector_id_nunique_-11','hist_merchant_id_count_-1','hist_category_1_mean','new_purchase_date_max','new_installments_max_2','new_month_lag_mean','new_merch_avg_sales_lag12_sum','hist_installments_sum_-4','hist_installments_max_-10','diff_purchase_by_merchant','new_purchase_date_diff','new_purchase_amount_min_2','hist_installments_mean_-5','new_purchase_month_min','hist_installments_mean_-6','diff_purchase_sum_1_0','new_purchase_amount_mean_2','hist_installments_max_0','hist_merch_most_recent_sales_range_mean','hist_month_diff_mean','hist_card_id_size','hist_installments_mean_-12','hist_purchase_amount_min_-1','hist_purchase_date_max','feature_3','hist_installments_std','diff_merchant_id_nunique_4_3','diff_purchase_sum_5_4','hist_installments_sum_-6','new_purchase_amount_std_1','feature_2','diff_subsector_id_lag_mean','hist_subsector_id_nunique_-6','hist_purchase_amount_max_-5','purchase_amount_sum_-6','hist_purchase_amount_mean_-2','hist_merch_tran_ratio','purchase_amount_sum_-13','hist_purchase_amount_min_-8','hist_installments_std_-4','hist_merch_numerical_2_mean','hist_installments_sum_-5','hist_subsector_id_nunique_-10','new_merch_most_recent_sales_range_sum','hist_purchase_amount_max_-8','hist_year_nunique','new_month_diff_nunique','hist_merch_avg_sales_lag6_max','hist_purchase_amount_mean_-1','hist_purchase_amount_min_-6','new_purchase_amount_max_1','hist_purchase_amount_std_-1','hist_merchant_category_id_nunique_-3','hist_purchase_amount_mean_-4','purchase_amount_sum_-4','hist_month_nunique','hist_merch_category_4_sum','new_purchase_amount_mean','new_purchase_month_max', 'hist_purchase_amount_std_-5']

    X=train[cols]    
    y=train['target']    
    X_test = test[cols]
   
    exp_id = randint(0, 10000)
    params = {
    'boosting_type': 'gbdt', 
    'colsample_bytree': 0.5170233956444398, 
    'learning_rate': 0.01143545759270505, 
    'min_child_weight': 200, 
    'min_split_gain': 0.24720655947734432, 
    'n_estimators': 8700, 
    'num_leaves': 31, 
    'reg_alpha': 0.14446986536955275, 
    'reg_lambda': 0.8821246705659376, 
    'subsample': 0.9444712859527173, 
    'subsample_for_bin': 140000.0}
    
    train_XGB(X, y, params, 5, False, exp_id, True, X_test)
    #test_pred = pd.read_csv('./data/interim/test_pred.csv')
    #pd.DataFrame({'card_id':test['card_id'], 'target':test_pred['test_pred']}).to_csv('./data/processed/submission_10.csv', index=False)    
		  
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()