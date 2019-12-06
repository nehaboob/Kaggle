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
import random

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
def lgbm_hyperopt_training(X, y):

	def objective(params):
	    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
	    
	    # Keep track of evals
	    global ITERATION
	    
	    ITERATION += 1
	    exp_id = randint(0, 10000)

	    # Make sure parameters that need to be integers are integers - subsample_freq
	    for parameter_name in ['num_leaves', 'min_child_weight', 'n_estimators']:
	        params[parameter_name] = int(params[parameter_name])
	    
	    start = timer()
	    
	    # set params
	    lgb_params = params
	    # Perform n_folds cross validation
	    cv_results = train_XGB(X,y,lgb_params,5,False,exp_id)
	    
	    run_time = timer() - start

	    # Dictionary with information for evaluation
	    info =  {'loss': cv_results, 'params': params, 'iteration': ITERATION, 
	            'train_time': run_time, 'status': STATUS_OK}

	    return info

	#'subsample_freq': hp.quniform('subsample_freq', 0, 100, 5),
       
	# Define the search space
	space = {
	    'boosting_type': 'gbdt',
	    'subsample': hp.uniform('subsample', 0.1, 1),
	    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 400000, 10000),
	    'n_estimators': 10000,
	    'num_leaves': hp.quniform('num_leaves', 20, 50, 5),
	    'max_depth':hp.choice('max_depth', [3, 4, 5, 6]),
	    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.025)),
	    'min_child_weight': hp.quniform('min_child_weight', 50, 150, 5),
	    'min_split_gain': hp.uniform('min_split_gain', 0.1, 1.0),
	    #'reg_alpha': hp.uniform('reg_alpha', 0.0001, 1.0),
	    #'reg_lambda': hp.uniform('reg_lambda', 0.0001, 1.0),
	    'colsample_bytree': hp.uniform('colsample_by_tree', 0.1, 1)
	}
		
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
		with open('./data/interim/train_'+str(i)+'.pickle', 'rb') as f:
			train_index = pickle.load(f)

		with open('./data/interim/val_'+str(i)+'.pickle', 'rb') as f:
			val_index = pickle.load(f)
		 

		model = lgb.LGBMRegressor(**params)
		model.fit(
                X.iloc[train_index], y.iloc[train_index],
                eval_set=[(X.iloc[val_index], y.iloc[val_index])],
                eval_metric='rmse',
                early_stopping_rounds=200,
                verbose=200
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
			test_pred = test_pred+(model.predict(X_test)/splits)
	
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

	if store_results:	    
		return scores_df['test_rmse'], oof_reg_preds, test_pred  
	else:
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
    exp_predictions_stk = pd.DataFrame()
    test_predictions_stk = pd.DataFrame()
    param_list=[{'boosting_type': 'gbdt', 'colsample_bytree': 0.6747020878192473, 'learning_rate': 0.01028946844855269, 'max_depth': 8, 'min_child_weight': 125, 'min_split_gain': 0.4478927896823448, 'n_estimators': 5000, 'num_leaves': 44, 'subsample': 0.7664331423300333, 'subsample_for_bin': 300000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.655354421667673, 'learning_rate': 0.012602456087666471, 'max_depth': -1, 'min_child_weight': 65, 'min_split_gain': 0.560545642457214, 'n_estimators': 5000, 'num_leaves': 50, 'subsample': 0.8995581655934168, 'subsample_for_bin': 120000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.5146962995407758, 'learning_rate': 0.012362408257744631, 'max_depth': 7, 'min_child_weight': 145, 'min_split_gain': 0.8645744164929342, 'n_estimators': 5000, 'num_leaves': 55, 'subsample': 0.739420658143976, 'subsample_for_bin': 300000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.5591500523406365, 'learning_rate': 0.008174653532391328, 'max_depth': 9, 'min_child_weight': 70, 'min_split_gain': 0.751068907542556, 'n_estimators': 8000, 'num_leaves': 50, 'subsample': 0.7466608071515193, 'subsample_for_bin': 320000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.5328640436513024, 'learning_rate': 0.007291193612704089, 'max_depth': 8, 'min_child_weight': 80, 'min_split_gain': 0.9430637956315739, 'n_estimators': 8000, 'num_leaves': 60, 'subsample': 0.7206327909797919, 'subsample_for_bin': 190000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.5001361247827644, 'learning_rate': 0.008174653532391328, 'max_depth': 9, 'min_child_weight': 145, 'min_split_gain': 0.680799532489138, 'n_estimators': 10000, 'num_leaves': 80, 'subsample': 0.7391096772075721, 'subsample_for_bin': 30000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.44006603718821186, 'learning_rate': 0.008174653532391328, 'max_depth': 9, 'min_child_weight': 90, 'min_split_gain': 0.8258820700989176, 'n_estimators': 10000, 'num_leaves': 80, 'subsample': 0.8857279206345009, 'subsample_for_bin': 100000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.5170233956444398, 'learning_rate': 0.01143545759270505, 'min_child_weight': 200, 'min_split_gain': 0.24720655947734432, 'n_estimators': 8700, 'num_leaves': 31, 'reg_alpha': 0.14446986536955275, 'reg_lambda': 0.8821246705659376, 'subsample': 0.9444712859527173, 'subsample_for_bin': 140000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.648160438571269, 'learning_rate': 0.010851387576828717, 'min_child_weight': 145, 'min_split_gain': 0.11603431271605252, 'n_estimators': 10700, 'num_leaves': 50, 'reg_alpha': 0.5616013112843976, 'reg_lambda': 0.3801503947767616, 'subsample': 0.9811948375658974, 'subsample_for_bin': 260000.0},
				{'boosting_type': 'gbdt', 'colsample_bytree': 0.4489950302629293, 'learning_rate': 0.015340569780328632, 'min_child_weight': 180, 'min_split_gain': 0.3426265245024095, 'n_estimators': 8400, 'num_leaves': 38, 'reg_alpha': 0.14619983278367954, 'reg_lambda': 0.009338268516283665, 'subsample': 0.9818525466651628, 'subsample_for_bin': 180000.0}]
    
   
    for i in range(0,20):
	    cols = ['new_category_1_sum','new_category_1_mean','new_subsect_purchase_mean_ratio','dayofweek','new_purchase_mean_month_diff','new_merch_purchase_mean_ratio','hist_merchant_id_nunique_-9','hist_merch_month_diff','hist_weekend_mean','purchase_amount_sum_-7','hist_purchase_amount_max_-2','feature_1','new_category_2_5.0_mean','diff_purchase_sum_6_5','hist_purchase_amount_min_0','new_subsector_id_nunique_1','new_purchase_amount_max_2','hist_first_buy','hist_merchant_id_count_-9','merch_purchase_diff_5_6','hist_purchase_amount_max','hist_purchase_amount_max_-1','new_merch_avg_sales_lag3_std','new_purchase_amount_min_1','hist_purchase_date_uptonow','diff_purchase_lag_mean','hist_installments_sum_0','hist_subsector_id_nunique_0','diff_purchase_sum_7_6','purchase_amount_sum_0','new_merch_numerical_2_min','new_merchant_category_id_nunique','hist_authorized_flag_mean','new_purchase_date_min','hist_purchase_amount_min_-4','hist_category_3_B_sum','new_purchase_month_mean','merch_purchase_diff_lag_mean','merch_purchase_diff_4_5','new_month_diff_mean','hist_purchase_amount_std_-4','hist_month_lag_nunique','hist_purchase_amount_mean_-3','new_purchase_date_ptp_ptp','hist_purchase_amount_mean_0','hist_merch_avg_purchases_lag3_max','hist_purchase_amount_sum_-5','hist_merchant_id_nunique_-3','new_purchase_amount_std','purchase_amount_total','purchase_amount_sum_-3','diff_subsector_id_nunique_5_4','hist_merchant_category_id_nunique_-7','purchase_amount_sum_-5','hist_subsector_id_nunique_-9','hist_purchase_amount_std_-3','hist_merchant_category_id_nunique_-5','diff_purchase_sum_9_8','hist_purchase_amount_min','hist_purchase_month_std','hist_category_1_sum','new_purchase_amount_max','hist_purchase_amount_max_0','hist_merchant_id_nunique_-10','hist_purchase_amount_sum_-4','diff_purchase_sum_8_7','hist_purchase_amount_max_-3','hist_installments_mean_0','hist_purchase_amount_mean_-6','hist_installments_mean_-11','hist_category_2_1.0_mean','hist_installments_mean_-4','diff_merchant_id_nunique_1_0','diff_purchase_sum_10_9','new_first_buy','hist_purchase_date_average','hist_merch_avg_purchases_lag12_max','new_merch_purchase_sum_ratio','hist_purchase_amount_mean_-5','new_city_id_nunique','hist_purchase_amount_min_-5','hist_merch_purchase_sum_ratio','hist_merchant_id_nunique_-4','hist_merchant_category_id_nunique_-9','hist_merch_purchase_mean_ratio','hist_purchase_date_min','hist_purchase_amount_min_-3','hist_merch_avg_sales_lag3_mean','hist_installments_sum','hist_installments_mean_-3','hist_purchase_amount_sum_-1','new_merch_avg_sales_lag6_sum','new_purchase_date_uptonow','hist_installments_mean_-8','hist_installments_std_-6','hist_purchase_amount_sum_-10','hist_purchase_amount_sum_0','elapsed_time','new_category_3_A_sum','new_purchase_date_average','hist_purchase_amount_max_-4','hist_category_3_C_mean','hist_subsector_id_nunique_-11','hist_merchant_id_count_-1','hist_category_1_mean','new_purchase_date_max','new_installments_max_2','new_month_lag_mean','new_merch_avg_sales_lag12_sum','hist_installments_sum_-4','hist_installments_max_-10','diff_purchase_by_merchant','new_purchase_date_diff','new_purchase_amount_min_2','hist_installments_mean_-5','new_purchase_month_min','hist_installments_mean_-6','diff_purchase_sum_1_0','new_purchase_amount_mean_2','hist_installments_max_0','hist_merch_most_recent_sales_range_mean','hist_month_diff_mean','hist_card_id_size','hist_installments_mean_-12','hist_purchase_amount_min_-1','hist_purchase_date_max','feature_3','hist_installments_std','diff_merchant_id_nunique_4_3','diff_purchase_sum_5_4','hist_installments_sum_-6','new_purchase_amount_std_1','feature_2','diff_subsector_id_lag_mean','hist_subsector_id_nunique_-6','hist_purchase_amount_max_-5','purchase_amount_sum_-6','hist_purchase_amount_mean_-2','hist_merch_tran_ratio','purchase_amount_sum_-13','hist_purchase_amount_min_-8','hist_installments_std_-4','hist_merch_numerical_2_mean','hist_installments_sum_-5','hist_subsector_id_nunique_-10','new_merch_most_recent_sales_range_sum','hist_purchase_amount_max_-8','hist_year_nunique','new_month_diff_nunique','hist_merch_avg_sales_lag6_max','hist_purchase_amount_mean_-1','hist_purchase_amount_min_-6','new_purchase_amount_max_1','hist_purchase_amount_std_-1','hist_merchant_category_id_nunique_-3','hist_purchase_amount_mean_-4','purchase_amount_sum_-4','hist_month_nunique','hist_merch_category_4_sum','new_purchase_amount_mean','new_purchase_month_max', 'hist_purchase_amount_std_-5']
	    num_col = np.random.randint(0, len(cols))
	    cols = random.sample(cols, num_col)
	    X=train[cols]    
	    y=train['target']    
	    X_test = test[cols]
	    print(X_test.shape)
	   	
	    exp_id = randint(0, 10000)
	    params = random.choice(param_list)
	    cv_score, oof_preds, test_preds = train_XGB(X, y, params, 5, True, exp_id, True, X_test)
	    test_pred = pd.read_csv('./data/interim/test_pred.csv')
	    exp_predictions_stk[str(exp_id)] = oof_preds
	    test_predictions_stk[str(exp_id)] = test_preds
	
    exp_predictions_stk.to_csv('./data/interim/exp_predictions_stk_15.csv', index=False)
    test_predictions_stk.to_csv('./data/interim/test_predictions_stk_15.csv', index=False)    
    '''

    exp_predictions_stk = pd.read_csv('./data/interim/exp_predictions_stk_1.csv')
    y=train['target'] 
    lgbm_hyperopt_training(exp_predictions_stk, y)

    #test_predictions_stk = pd.read_csv('./data/interim/test_predictions_stk_1.csv')  
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