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


################################################
# hyper param tuning
################################################
def lgbm_hyperopt_training(X, y, feats):

	def objective(params):
	    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
	    
	    # Keep track of evals
	    global ITERATION
	    
	    ITERATION += 1
	    exp_id = randint(0, 10000)

	    # Retrieve the subsample if present otherwise set to 1.0
	    #subsample = params['boosting_type'].get('subsample', 1.0)
	    
	    # Extract the boosting type
	    #params['boosting_type'] = params['boosting_type']['boosting_type']
	    #params['subsample'] = subsample
	    
	    # Make sure parameters that need to be integers are integers - subsample_freq
	    for parameter_name in ['num_leaves', 'min_child_weight', 'n_estimators', 'num_feats']:
	        params[parameter_name] = int(params[parameter_name])
	    
	    start = timer()
	    
	    # set params
	    print(params)
	    num_feats = params['num_feats']
	    del params['num_feats']
	    lgb_params = params
	    print('features', num_feats)
	    x_interim = X[feats[:num_feats]]
	    # Perform n_folds cross validation
	    cv_results = train_XGB(x_interim,y,lgb_params,5,False,exp_id)
	    
	    run_time = timer() - start

	    # Dictionary with information for evaluation
	    info =  {'loss': cv_results, 'params': params, 'iteration': ITERATION, 
	            'train_time': run_time, 'status': STATUS_OK}

	    return info

	#'subsample_freq': hp.quniform('subsample_freq', 0, 100, 5),
       
	# Define the search space
	space = {
		'num_feats': hp.quniform('num_feats', 30, 200, 20),
	    'boosting_type': 'gbdt',
	    'subsample': hp.uniform('subsample', 0.1, 1),
	    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 10000),
	    'n_estimators': hp.quniform('n_estimators', 3000, 15000, 100),
	    'num_leaves': hp.quniform('num_leaves', 30, 100, 1),
	    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.4)),
	    'min_child_weight': hp.quniform('min_child_weight', 20, 200, 5),
	    'min_split_gain': hp.uniform('min_split_gain', 0.001, 1.0),
	    'reg_alpha': hp.uniform('reg_alpha', 0.0001, 1.0),
	    'reg_lambda': hp.uniform('reg_lambda', 0.0001, 1.0),
	    'colsample_bytree': hp.uniform('colsample_by_tree', 0.05, 1.0)
	}
		
	tpe_algorithm = tpe.suggest
	bayes_trials = Trials()
	global  ITERATION

	ITERATION = 0
	MAX_EVALS = 50

	# Run optimization , rstate = np.random.RandomState(50)
	best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

	# Sort the trials with lowest loss (highest AUC) first
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])

	# train XGB with best results and store the model
	#train_XGB(X,y,lgb_params, store_results=True)
	print(bayes_trials_results[:2])
	print(best)


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
	for i in range(splits):
		with open('./data/interim/train_skf_'+str(i)+'.pickle', 'rb') as f:
			train_index = pickle.load(f)

		with open('./data/interim/val_skf_'+str(i)+'.pickle', 'rb') as f:
			val_index = pickle.load(f)
		 

		model = lgb.LGBMRegressor(**params)
		model.fit(
                X.iloc[train_index], y.iloc[train_index],
                eval_set=[(X.iloc[val_index], y.iloc[val_index])],
                eval_metric='rmse',
                early_stopping_rounds=100,
                #categorical_feature=['feature_1', 'feature_2', 'feature_3'],
                verbose=False
            )

		# predict train and val for getting scores
		y_true = y.iloc[val_index]
		y_pred = model.predict(X.iloc[val_index])
		oof_reg_preds[val_index] = y_pred

		y_train_true = y.iloc[train_index]
		y_train_pred = model.predict(X.iloc[train_index])

		tmp_train_score = get_scores(y_train_true, y_train_pred,'train')
		if(i==0):
			train_scores = tmp_train_score
		else:
			train_scores = train_scores+tmp_train_score

		val_score = get_scores(y_true, y_pred, 'test')		
		feature_imp = feature_imp+pd.Series(index= X.columns.values, data= model.feature_importances_)
		log_metrics(val_score, tmp_train_score, feature_imp.sort_values(ascending=False).head(20), params, i, exp_id)
		if(test_results):
			test_pred = test_pred+(model.predict(X_test)/5)
	
	feature_imp.sort_values(ascending=False).to_csv('features.csv')
	#print(feature_imp.sort_values(ascending=False)[:100])
	if(test_results):
		pd.DataFrame({'test_pred': test_pred}).to_csv('./data/interim/test_pred.csv', index=False)
	
	#store oof predictions
	#pd.DataFrame({'true_vals': y.values, 'predictions':oof_reg_preds}).to_csv('oof_predictions_20170501.csv', index=False)
	scores_df = get_scores(y, oof_reg_preds,'test')
	print("test RMSE - ", scores_df['test_rmse'], "train RMSE - ", train_scores['train_rmse']/splits)

	run_time = timer() - start

	exp_desc, algo = get_my_config()
	with open('results_exp.csv', 'a') as f:
		    spamwriter = csv.writer(f)
		    spamwriter.writerow([		    
		    	exp_id,
		    	exp_desc,
		    	algo,
		    	params,
		    	scores_df['test_rmse'],
		    	train_scores['train_rmse']/splits,
		    	run_time])

	return scores_df['test_rmse']                  


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
    feats = ['hist_purchase_date_max','new_purchase_date_uptonow','hist_purchase_date_uptonow','new_purchase_month_mean','hist_authorized_flag_mean','new_purchase_date_max','new_purchase_amount_max','hist_category_1_sum','new_month_diff_mean','new_purchase_date_min','hist_category_1_mean','hist_month_diff_mean','new_purchase_date_ptp_ptp','new_merch_purchase_mean_ratio','new_purchase_date_average','new_subsect_purchase_mean_ratio','new_purchase_mean_month_diff','hist_first_buy','new_category_1_mean','hist_merch_tran_ratio','new_purchase_amount_std','purchase_amount_sum_0','hist_installments_sum_0','new_purchase_amount_mean','hist_merch_month_diff','hist_purchase_date_min','new_purchase_date_diff','hist_purchase_month_std','new_month_lag_mean','hist_month_lag_nunique','new_purchase_month_max','hist_purchase_amount_max_-2','hist_purchase_amount_max_-1','hist_purchase_amount_max_0','new_purchase_month_min','purchase_amount_sum_-4','hist_purchase_amount_mean_0','merch_purchase_diff_4_5','diff_purchase_sum_6_5','new_category_1_sum','hist_installments_sum_-4','new_purchase_amount_max_2','diff_purchase_by_merchant','merch_purchase_diff_5_6','hist_installments_mean_-6','hist_purchase_amount_std_-5','hist_installments_mean_-4','hist_merch_purchase_sum_ratio','merch_purchase_diff_lag_mean','new_purchase_amount_mean_2','hist_purchase_amount_max_-3','elapsed_time','hist_purchase_amount_min_-4','hist_purchase_amount_mean_-5','hist_merch_purchase_mean_ratio','hist_purchase_amount_max_-5','purchase_amount_sum_-5','hist_purchase_amount_min_-1','new_purchase_amount_max_1','hist_purchase_date_average','hist_merch_category_4_sum','hist_purchase_amount_min_-6','hist_month_nunique','hist_installments_sum','hist_purchase_amount_sum_-4','hist_purchase_amount_std_-4','purchase_amount_sum_-6','hist_purchase_amount_min_0','hist_installments_std_-4','hist_installments_sum_-6','hist_purchase_amount_std_-3','hist_installments_mean_0','hist_category_3_C_mean','diff_purchase_sum_9_8','hist_installments_mean_-5','hist_purchase_amount_sum_-5','new_merch_purchase_sum_ratio','diff_purchase_sum_1_0','diff_purchase_lag_mean','hist_purchase_amount_min_-5','hist_purchase_amount_min_-3','new_purchase_amount_min_2','new_purchase_amount_std_1','hist_category_2_1.0_mean','hist_purchase_amount_min','purchase_amount_sum_-7','diff_purchase_sum_8_7','hist_year_nunique','diff_purchase_sum_5_4','hist_purchase_amount_max_-4','diff_purchase_sum_7_6','hist_purchase_amount_mean_-1','hist_purchase_amount_mean_-3','hist_purchase_amount_sum_0','hist_purchase_amount_max','hist_installments_sum_-5','hist_installments_std','feature_1','hist_purchase_amount_mean_-4','new_purchase_amount_min_1','purchase_amount_sum_-2','hist_purchase_amount_mean_-6','hist_purchase_amount_mean_-2','purchase_amount_sum_-1','hist_category_3_B_sum','purchase_amount_sum_-3','hist_purchase_amount_std_-1','hist_installments_mean','hist_purchase_amount_sum_-6','hist_purchase_amount_std','hist_purchase_amount_std_-6','new_first_buy','new_purchase_amount_sum_2','diff_purchase_sum_3_2','hist_installments_std_-5', 'hist_merch_most_recent_purchases_range_std','merch_purchase_diff_3_4','merch_purchase_diff_0_1','hist_purchase_amount_max_-6','diff_purchase_sum_4_3','hist_purchase_amount_min_-7','hist_purchase_amount_std_-2','hist_purchase_amount_std_0','purchase_amount_sum_-8','hist_purchase_amount_min_-8','hist_purchase_amount_sum_-1','hist_purchase_amount_max_-7', 'diff_purchase_by_subsector','hist_weekend_mean','new_installments_max','diff_purchase_sum_2_1','hist_category_2_1.0_sum','merch_purchase_diff_1_2','hist_merch_most_recent_purchases_range_mean','hist_installments_mean_-3','hist_weekofyear_nunique','hist_purchase_amount_max_-8','hist_purchase_amount_std_-8','hist_purchase_amount_max_-9','hist_merch_active_months_lag12_std','diff_merchant_id_nunique_5_4','diff_purchase_sum_10_9','hist_purchase_amount_mean_-7','hist_month_lag_std','new_purchase_amount_mean_1','hist_merchant_id_count_0','hist_purchase_amount_std_-7','new_purchase_amount_min','hist_merch_category_4_mean','hist_installments_std_-7','hist_purchase_amount_sum_-3','hist_installments_std_-6','new_merch_avg_purchases_lag12_min','hist_purchase_amount_sum_-8','hist_merch_avg_purchases_lag6_min','hist_installments_sum_-3','merch_purchase_diff_2_3','hist_installments_mean_-7','hist_purchase_mean_month_diff','hist_purchase_amount_min_-2','hist_merch_most_recent_sales_range_std','new_purchase_month_std','hist_purchase_amount_std_-9','hist_subsect_purchase_mean_ratio','hist_installments_sum_-8','hist_installments_std_-8','hist_category_3_C_sum','hist_merch_avg_purchases_lag3_min','hist_merch_avg_purchases_lag3_std','hist_category_3_B_mean','hist_purchase_amount_mean_-8','hist_installments_mean_-1','hist_merch_avg_sales_lag12_min','hist_installments_std_-3','hist_purchase_amount_sum_-2','new_purchase_sum_month_diff','hist_merch_numerical_2_std','hist_installments_sum_-2','hist_purchase_amount_mean','hist_merch_avg_purchases_lag3_sum','new_subsect_purchase_sum_ratio','new_merch_avg_purchases_lag3_mean','hist_purchase_month_mean','hist_installments_std_-1','hist_purchase_date_ptp_ptp','hist_month_diff_nunique','hist_merch_avg_sales_lag3_std','hist_merch_most_recent_sales_range_mean','new_merch_most_recent_purchases_range_mean','hist_subsect_tran_ratio','new_installments_mean','new_merch_avg_purchases_lag6_sum','feature_2','new_purchase_amount_std_2','purchase_amount_sum_-9','hist_merch_numerical_1_std','new_merch_avg_purchases_lag6_min','diff_merchant_id_nunique_1_0','hist_purchase_date_diff','diff_subsector_id_nunique_5_4','hist_purchase_amount_sum_-7','diff_subsector_id_lag_mean','diff_merchant_id_nunique_3_2','new_merch_avg_sales_lag12_min','hist_installments_std_0','diff_subsector_id_nunique_1_0','hist_installments_sum_-9','new_merch_avg_purchases_lag3_min','diff_merchant_id_lag_mean','new_merch_numerical_1_min','new_merch_avg_purchases_lag3_max','new_installments_sum','hist_purchase_month_min','new_merch_avg_purchases_lag12_sum','hist_merchant_id_count_-5','new_merch_avg_sales_lag3_std','hist_installments_sum_-1','new_merchant_category_id_nunique_1','hist_installments_max_-4','new_merch_avg_purchases_lag12_max','hist_merch_numerical_2_sum','new_purchase_amount_sum','hist_merch_avg_purchases_lag12_min','hist_merch_avg_sales_lag6_min','hist_merchant_category_id_nunique_-3','diff_subsector_id_nunique_4_3','hist_installments_mean_-2','hist_purchase_amount_mean_-9','hist_installments_mean_-11','hist_subsector_id_nunique_0','hist_merch_avg_purchases_lag3_mean','new_merch_avg_purchases_lag3_std','hist_installments_mean_-8','diff_purchase_sum_12_11','hist_installments_sum_-7','new_purchase_amount_sum_1','hist_merch_subsect_ratio','hist_merch_avg_purchases_lag6_sum','hist_weekend_sum','hist_merchant_id_nunique_-5','hist_purchase_amount_max_-11','hist_merchant_id_count_-4','new_merch_numerical_2_std','hist_merchant_id_count_-6','hist_merch_avg_sales_lag3_mean','hist_merchant_id_nunique_0','new_merch_avg_sales_lag3_mean','diff_merchant_id_nunique_4_3','new_merch_avg_purchases_lag3_sum','hist_category_3_A_sum','new_merch_avg_sales_lag6_min','hist_merch_avg_sales_lag3_sum','hist_merch_avg_sales_lag3_min','month','new_merch_avg_sales_lag3_sum','hist_merch_numerical_1_sum','hist_purchase_amount_min_-9','hist_subsect_purchase_sum_ratio','diff_purchase_sum_11_10','new_category_3_A_sum','hist_merchant_id_nunique_-4','hist_merchant_id_nunique_-2','new_merch_numerical_2_mean','new_merch_avg_sales_lag12_std','weekofyear','hist_merch_most_recent_sales_range_sum','new_merch_avg_purchases_lag6_std','purchase_amount_total','new_merch_avg_sales_lag3_min','hist_installments_max_0','hist_purchase_amount_sum_-9','hist_merchant_category_id_nunique_0','hist_merch_active_months_lag12_mean','hist_purchase_month_max','hist_installments_min_0','new_merch_avg_sales_lag12_sum','hist_merch_avg_purchases_lag12_sum','new_merch_avg_sales_lag6_max','hist_merch_avg_sales_lag6_std','new_merch_avg_purchases_lag6_max','new_merch_most_recent_sales_range_mean','new_category_3_B_mean','new_merch_numerical_1_mean','new_merchant_id_nunique_1','new_merch_avg_sales_lag12_mean','hist_purchase_amount_sum','hist_installments_max_-2','hist_merch_avg_sales_lag12_mean','purchase_amount_sum_-10','hist_merch_avg_sales_lag12_std','hist_month_lag_max','hist_merchant_id_count_-2','hist_installments_std_-9','new_merch_most_recent_purchases_range_std','hist_merchant_id_count_-8','hist_purchase_amount_max_-10','hist_merch_avg_sales_lag6_sum','hist_purchase_sum_month_diff','new_merch_avg_sales_lag6_sum','new_merch_avg_sales_lag6_mean','new_merch_category_4_mean','hist_merch_numerical_1_mean','hist_merchant_id_count_-1','new_merch_avg_purchases_lag6_mean','new_month_lag_std','hist_installments_mean_-9','new_merch_avg_sales_lag6_std','hist_merch_avg_purchases_lag12_std','new_merch_avg_purchases_lag12_mean','hist_merch_avg_purchases_lag6_std','diff_subsector_id_nunique_3_2','new_installments_min','new_merch_avg_sales_lag3_max','hist_installments_max_-6','hist_installments_std_-2','new_subsector_id_nunique_1','hist_merch_most_recent_purchases_range_sum','hist_merch_numerical_2_mean','hist_category_3_A_mean','purchase_amount_sum_-11','new_merch_avg_purchases_lag12_std','hist_purchase_amount_mean_-10','hist_subsector_id_nunique_-5','new_merchant_id_count_1','hist_installments_max_-5','diff_merchant_id_nunique_2_1','new_merch_numerical_1_std','new_merch_avg_sales_lag12_max','hist_merch_merchant_group_id_nunique','hist_purchase_amount_mean_-11','hist_merchant_id_nunique_-7','hist_merch_avg_sales_lag6_mean','hist_hour_nunique','hist_purchase_amount_min_-11','hist_merchant_id_nunique_-3','hist_merchant_id_nunique_-1','hist_merch_numerical_1_max','hist_installments_max','hist_merch_avg_sales_lag12_sum','card_id_total','new_merch_numerical_1_sum','hist_merchant_category_id_nunique_-5','new_merch_numerical_2_max','hist_installments_max_-7','hist_merch_avg_purchases_lag12_mean','hist_authorized_flag_sum','new_merch_numerical_1_max','hist_purchase_amount_std_-10','hist_merchant_id_count_-7','new_installments_sum_1','diff_purchase_sum_13_12','hist_subsector_id_nunique_-3','hist_subsector_id_nunique_-1','hist_category_2_3.0_mean','hist_purchase_amount_min_-10','hist_purchase_amount_std_-11','hist_merchant_id_nunique_-6','hist_category_2_2.0_mean','hist_purchase_amount_sum_-11','hist_merchant_category_id_nunique','diff_subsector_id_nunique_2_1','dayofweek','hist_merchant_category_id_nunique_-2','feature_3','hist_merchant_category_id_nunique_-6','new_merch_most_recent_sales_range_sum','hist_merchant_id_count_-3','hist_subsector_id_nunique_-6','hist_installments_min_-5','new_merch_numerical_2_sum','hist_installments_std_-10','hist_merchant_category_id_nunique_-4','new_merch_most_recent_sales_range_std','hist_merch_avg_purchases_lag6_mean','hist_subsector_id_nunique','hist_purchase_amount_std_-12','hist_purchase_amount_mean_-12','new_installments_mean_1','hist_category_2_4.0_sum','hist_installments_mean_-10','new_merch_most_recent_purchases_range_sum','new_category_3_B_sum','hist_merchant_category_id_nunique_-1','hist_merch_active_months_lag12_sum','hist_installments_max_-3','hist_merchant_id_nunique','hist_subsector_id_nunique_-4','hist_merch_active_months_lag6_std','new_weekend_mean','hist_merchant_id_count_-9','hist_installments_min_-11','hist_installments_min_-7','hist_installments_max_-8','new_merch_category_4_sum','hist_category_2_3.0_sum','new_merch_numerical_2_min','hist_purchase_amount_sum_-10','hist_purchase_amount_max_-12','hist_merch_numerical_2_max','hist_category_2_4.0_mean','hist_merch_avg_purchases_lag6_max','hist_merch_avg_sales_lag12_max','hist_merchant_category_id_nunique_-7','hist_subsector_id_nunique_-7','hist_merch_avg_purchases_lag12_max','hist_installments_sum_-10','hist_installments_std_-11','hist_installments_mean_-12','hist_merch_avg_purchases_lag3_max','new_installments_mean_2','hist_category_2_2.0_sum','hist_city_id_nunique','hist_merch_avg_sales_lag6_max','hist_category_2_5.0_mean','new_merchant_id_nunique_2','hist_merchant_id_nunique_-8','new_merch_month_diff','new_installments_std','hist_merch_avg_sales_lag3_max','new_merch_most_recent_sales_range_min','new_merch_most_recent_purchases_range_min','hist_installments_min_-4','new_category_2_1.0_sum','hist_merchant_category_id_nunique_-8','new_merch_subsect_ratio','hist_state_id_nunique','hist_installments_max_-1','hist_subsector_id_nunique_-8','hist_installments_sum_-11','new_category_2_5.0_mean','hist_subsector_id_nunique_-2','new_category_2_1.0_mean','new_subsect_tran_ratio','hist_merchant_id_nunique_-9','hist_card_id_size','purchase_amount_sum_-12','new_installments_max_1','hist_merch_active_months_lag3_sum','new_merchant_category_id_nunique_2','hist_category_2_5.0_sum','hist_installments_max_-10','hist_installments_min_-3','new_merch_active_months_lag12_sum','hist_purchase_amount_min_-12','hist_installments_max_-9','new_category_3_C_mean','new_merch_active_months_lag12_mean','hist_installments_min_-8','new_month_diff_nunique','new_installments_sum_2','new_merchant_id_count_2','hist_installments_min','new_subsector_id_nunique_2','new_weekofyear_nunique','new_weekend_sum','hist_subsector_id_nunique_-9','hist_installments_min_-2','hist_installments_min_-6','new_installments_max_2','new_merchant_category_id_nunique','hist_installments_min_-1','hist_purchase_amount_min_-13','hist_merchant_id_count_-10','hist_merchant_category_id_nunique_-9','new_installments_std_1','hist_purchase_amount_max_-13','hist_merch_active_months_lag12_min','new_month_lag_max','new_state_id_nunique','new_merch_most_recent_sales_range_max','new_merch_most_recent_purchases_range_max','new_installments_min_1','new_city_id_nunique','new_merch_active_months_lag12_std','new_category_2_4.0_sum','hist_purchase_amount_sum_-12','hist_merch_active_months_lag6_sum','hist_merchant_id_count_-11','new_category_2_2.0_mean','new_merch_merchant_group_id_nunique','new_category_3_A_mean','hist_dayofweek_nunique','hist_subsector_id_nunique_-10','hist_installments_max_-11','new_subsector_id_nunique','hist_merch_active_months_lag6_mean','new_category_2_4.0_mean','new_dayofweek_nunique','hist_purchase_amount_mean_-13','hist_merchant_category_id_nunique_-10','new_hour_nunique','purchase_amount_sum_-13','new_card_id_size','hist_merchant_id_nunique_-10','hist_merchant_category_id_nunique_-11','new_installments_min_2','hist_purchase_amount_std_-13','hist_purchase_amount_sum_-13','new_category_2_3.0_sum','new_year_nunique','hist_installments_min_-9','hist_merchant_category_id_nunique_-12','hist_merchant_id_count_-12','hist_installments_mean_-13','new_merch_tran_ratio','hist_installments_sum_-12','new_category_3_C_sum','new_category_2_3.0_mean','hist_merchant_id_nunique_-11','new_category_2_2.0_sum','new_authorized_flag_sum','new_category_2_5.0_sum','hist_subsector_id_nunique_-12','new_merch_active_months_lag12_min','new_installments_std_2','hist_merchant_id_count_-13','new_month_lag_nunique','new_month_nunique','new_month_lag_min','hist_merchant_id_nunique_-12','hist_installments_min_-10','new_merch_active_months_lag6_sum','hist_subsector_id_nunique_-11','hist_installments_std_-12','hist_subsector_id_nunique_-13','new_merchant_id_nunique','hist_merch_most_recent_purchases_range_max','hist_installments_std_-13','new_merch_active_months_lag6_mean','hist_merchant_category_id_nunique_-13','hist_installments_max_-12','new_merch_active_months_lag6_min','hist_merch_most_recent_sales_range_min','hist_installments_max_-13','hist_installments_min_-13','hist_merch_most_recent_sales_range_max','new_merch_active_months_lag3_sum','hist_installments_sum_-13','new_merch_active_months_lag12_max','hist_merch_most_recent_purchases_range_min','hist_merchant_id_nunique_-13','hist_merch_numerical_1_min','hist_merch_active_months_lag3_std','new_authorized_flag_mean','hist_installments_min_-12','hist_merch_active_months_lag12_max','new_merch_active_months_lag6_std','hist_merch_active_months_lag3_mean','new_merch_active_months_lag6_max','hist_merch_active_months_lag3_max','hist_merch_active_months_lag3_min','new_merch_active_months_lag3_std','hist_month_lag_min','new_merch_active_months_lag3_min', 'new_merch_active_months_lag3_max','new_merch_active_months_lag3_mean','hist_merch_active_months_lag6_max','hist_merch_active_months_lag6_min','hist_merch_numerical_2_min','hist_month_lag_mean','month_lag']
    y=train['target']
    lgbm_hyperopt_training(train, y, feats)

    '''
    #lgbm_hyperopt_training(X,y)
    #features selection
    for i in range(65, 85, 5):
	    print(i)
	    cols = feats[:i]    	
	    X=train[cols]
	    y=train['target']
	    X_test = test[cols]
   
	    exp_id = randint(0, 10000)
	    params = {'boosting_type': 'gbdt', 'colsample_bytree': 0.5170233956444398, 'learning_rate': 0.01143545759270505, 'min_child_weight': 200, 'min_split_gain': 0.24720655947734432, 'n_estimators': 8700, 'num_leaves': 31, 'reg_alpha': 0.14446986536955275, 'reg_lambda': 0.8821246705659376, 'subsample': 0.9444712859527173, 'subsample_for_bin': 140000.0}
	    train_XGB(X, y, params, 5, False, exp_id)
	    #train_XGB(X, y, params, 5, False, exp_id, True, X_test)
	    #test_pred = pd.read_csv('./data/interim/test_pred.csv')
	    #pd.DataFrame({'card_id':test['card_id'], 'target':test_pred['test_pred']}).to_csv('./data/processed/submission_9.csv', index=False)    
	'''

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()