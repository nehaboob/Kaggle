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
       
	# Define the search space
	space = {
	    'boosting_type': 'gbdt',
	    'subsample': hp.uniform('subsample', 0.1, 1),
	    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 400000, 10000),
	    'n_estimators': 10000,
	    'num_leaves': hp.quniform('num_leaves', 20, 50, 5),
	    'max_depth':hp.choice('max_depth', [3, 4, 5, 6, -1]),
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
	MAX_EVALS = 100

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
			test_pred = test_pred+model.predict(X_test)
	
	test_pred = test_pred/splits
	feature_imp.sort_values(ascending=False).to_csv('features.csv')
	#print(feature_imp.sort_values(ascending=False)[:100])
	if(test_results):
		pd.DataFrame({'test_pred': test_pred}).to_csv('./data/interim/test_pred.csv', index=False)
	
	#store oof predictions
	pd.DataFrame({'true_vals': y.values, 'predictions':oof_reg_preds}).to_csv('oof_predictions.csv', index=False)
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
    test=pd.read_csv('./data/processed/test_transactions_agg_v3.csv')
    test=test.fillna(0)

    exp_predictions_stk1 = pd.read_csv('./data/interim/exp_predictions_stk_1.csv')
    exp_predictions_stk2 = pd.read_csv('./data/interim/exp_predictions_stk_2.csv')
    exp_predictions_stk3 = pd.read_csv('./data/interim/exp_predictions_stk_3.csv')
    exp_predictions_stk4 = pd.read_csv('./data/interim/exp_predictions_stk_4.csv')
    exp_predictions_stk5 = pd.read_csv('./data/interim/exp_predictions_stk_5.csv')
    exp_predictions_stk6 = pd.read_csv('./data/interim/exp_predictions_stk_6.csv')
    exp_predictions_stk7 = pd.read_csv('./data/interim/exp_predictions_stk_7.csv')
    exp_predictions_stk8 = pd.read_csv('./data/interim/exp_predictions_stk_8.csv')
    exp_predictions_stk9 = pd.read_csv('./data/interim/exp_predictions_stk_9.csv')
    exp_predictions_stk10 = pd.read_csv('./data/interim/exp_predictions_stk_10.csv')    
    exp_predictions_stk11 = pd.read_csv('./data/interim/exp_predictions_stk_11.csv')
    exp_predictions_stk12 = pd.read_csv('./data/interim/exp_predictions_stk_12.csv')

    exp_predictions_stk = pd.concat([exp_predictions_stk1, exp_predictions_stk2, exp_predictions_stk3, 
    	exp_predictions_stk4, exp_predictions_stk5, exp_predictions_stk6, exp_predictions_stk7, exp_predictions_stk8,
    	exp_predictions_stk9, exp_predictions_stk10, exp_predictions_stk11, exp_predictions_stk12], axis=1)

    exp_predictions_stk['mean'] = exp_predictions_stk.mean(axis=1)
    y=train['target'] 
    print(exp_predictions_stk.shape)
    #lgbm_hyperopt_training(exp_predictions_stk, y)

    
    test_predictions_stk1 = pd.read_csv('./data/interim/test_predictions_stk_1.csv')  
    test_predictions_stk2 = pd.read_csv('./data/interim/test_predictions_stk_2.csv')  
    test_predictions_stk3 = pd.read_csv('./data/interim/test_predictions_stk_3.csv')  
    test_predictions_stk4 = pd.read_csv('./data/interim/test_predictions_stk_4.csv')  
    test_predictions_stk5 = pd.read_csv('./data/interim/test_predictions_stk_5.csv')  
    test_predictions_stk6 = pd.read_csv('./data/interim/test_predictions_stk_6.csv')  
    test_predictions_stk7 = pd.read_csv('./data/interim/test_predictions_stk_7.csv')  
    test_predictions_stk8 = pd.read_csv('./data/interim/test_predictions_stk_8.csv')  
    test_predictions_stk9 = pd.read_csv('./data/interim/test_predictions_stk_9.csv')  
    test_predictions_stk10 = pd.read_csv('./data/interim/test_predictions_stk_10.csv')  
    test_predictions_stk11 = pd.read_csv('./data/interim/test_predictions_stk_11.csv')  
    test_predictions_stk12 = pd.read_csv('./data/interim/test_predictions_stk_12.csv') 
 
    test_predictions_stk = pd.concat([test_predictions_stk1, test_predictions_stk2, test_predictions_stk3, 
    		test_predictions_stk4, test_predictions_stk5, test_predictions_stk6, test_predictions_stk7, test_predictions_stk8,
    		test_predictions_stk9, test_predictions_stk10, test_predictions_stk11, test_predictions_stk12], axis=1)
	

    exp_id = randint(0, 10000)
    test_predictions_stk['mean'] = test_predictions_stk.mean(axis=1)
    params={'boosting_type': 'gbdt', 'colsample_bytree': 0.8833846162363688, 'learning_rate': 0.02133075537776159, 'max_depth': 3, 'min_child_weight': 105, 'min_split_gain': 0.1942806586355228, 'n_estimators': 1000, 'num_leaves': 25, 'subsample': 0.3670028128957203, 'subsample_for_bin': 400000.0}
    
    '''
    model = lgb.LGBMRegressor(**params)    
    model.fit(
                exp_predictions_stk, y,
                eval_set=[(exp_predictions_stk, y)],
                eval_metric='rmse',
                verbose=100
            )
	
    test_preds = model.predict(test_predictions_stk)
    '''
    
    cv_score, oof_preds, test_preds = train_XGB(exp_predictions_stk, y, params, 5, True, exp_id, True, test_predictions_stk)    
    #pd.DataFrame({'card_id':test['card_id'], 'target':test_preds}).to_csv('./data/processed/submission_15.csv', index=False)    
	


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()