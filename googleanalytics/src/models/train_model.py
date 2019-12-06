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

# reg_lambda higher for overfitting
# reg_alpha higher for overfitting
global ITERATION, lgb_params, params_clf

def get_my_config(default=False):
	global ITERATION, lgb_params, params_clf
	if(default):
		lgb_params = {'boosting_type': 'dart', 'colsample_bytree': 0.4710970164512299, 'learning_rate': 0.031642623234827805, 'min_child_weight': 165, 'min_split_gain': 0.4251945969203937, 'n_estimators': 600, 'num_leaves': 138, 'reg_alpha': 0.5005519711106031, 'reg_lambda': 0.45127104117580963, 'subsample_for_bin': 260000.0, 'subsample': 0.3738991074449508} 
		
		params_clf = {
	        'n_estimators': 1000,
		    "max_bin": 512,
		    "learning_rate": 0.02,
		    "boosting_type": "gbdt",
		    "objective": "binary",
		    "metric": "binary_logloss",
		    "num_leaves": 10,
		    "min_data": 100,
		    "boost_from_average": True
		}

	exp_desc = 'Base version. Train Classifier and Regressor seperately.'
	algo = 'LGBM'
	return lgb_params, params_clf, exp_desc, algo

###################################
# logging and helper function
###################################
def cal_rmse(y_true, y_predict):
	rms = sqrt(mean_squared_error(y_true, y_predict))
	return rms

def get_scores(y_true, y_predict, y_predict_proba_or_reg, mode, algo_type):
    scores_df = pd.Series()

    if(algo_type == 'reg'):
    	scores_df[mode+'_'+algo_type+'_rmse'] = cal_rmse(y_true, y_predict_proba_or_reg)
    	scores_df[mode+'_'+algo_type+'_rmse_all_zero'] = cal_rmse(y_true, np.zeros_like(y_true))

    	y_predict = (y_predict_proba_or_reg > 0)*1
    	y_true = (y_true > 0)*1

    cm = confusion_matrix(y_true, y_predict)
    FP = cm[0][1]
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]

    scores_df[mode+'_'+algo_type+'_roc_auc'] = roc_auc_score(y_true, y_predict_proba_or_reg)
    scores_df[mode+'_'+algo_type+'_accuracy'] = accuracy_score(y_true, y_predict)
    scores_df[mode+'_'+algo_type+'_precision'] = precision_score(y_true, y_predict)
    scores_df[mode+'_'+algo_type+'_recall'] = recall_score(y_true, y_predict)
    scores_df[mode+'_'+algo_type+'_f1'] = f1_score(y_true, y_predict)
    scores_df[mode+'_'+algo_type+'_FPR'] = FP/(FP+TN)
    scores_df[mode+'_'+algo_type+'_TPR'] = TP/(TP+FN)
    scores_df[mode+'_'+algo_type+'_TN'] = TN
    scores_df[mode+'_'+algo_type+'_FP'] = FP
    scores_df[mode+'_'+algo_type+'_FN'] = FN
    scores_df[mode+'_'+algo_type+'_TP'] = TP

    return scores_df

def log_metrics(scores_df, train_scores, feature_importances):
    n = randint(0, 10000)
    lgb_params, params_clf, exp_desc, algo = get_my_config()

    with open('results_metrics.csv', 'a') as f:
    	spamwriter = csv.writer(f)
    	spamwriter.writerow([
    						n,
    						exp_desc,
                            algo,
                            scores_df['test_reg_rmse'],                  
							scores_df['test_reg_rmse_all_zero'],         
							scores_df['test_reg_roc_auc'],              
							scores_df['test_reg_accuracy'],              
							scores_df['test_reg_precision'],            
							scores_df['test_reg_recall'],                
							scores_df['test_reg_f1'],                   
							scores_df['test_reg_FPR'],                  
							scores_df['test_reg_TPR'],                  
							scores_df['test_reg_TN'],               
							scores_df['test_reg_FP'],                
							scores_df['test_reg_FN'],                   
							scores_df['test_reg_TP'], 
							scores_df['test_clf_roc_auc'], 
							scores_df['test_clf_accuracy'],   
							scores_df['test_clf_precision'],
							scores_df['test_clf_recall'],    
							scores_df['test_clf_f1'],               
							scores_df['test_clf_FPR'],              
							scores_df['test_clf_TPR'],           
							scores_df['test_clf_TN'],             
							scores_df['test_clf_FP'],              
							scores_df['test_clf_FN'],                  
							scores_df['test_clf_TP'],
							train_scores['train_reg_rmse'],                  
							train_scores['train_reg_rmse_all_zero'],         
							train_scores['train_reg_roc_auc'],              
							train_scores['train_reg_accuracy'],              
							train_scores['train_reg_precision'],            
							train_scores['train_reg_recall'],                
							train_scores['train_reg_f1'],                   
							train_scores['train_reg_FPR'],                  
							train_scores['train_reg_TPR'],                  
							train_scores['train_reg_TN'],               
							train_scores['train_reg_FP'],                
							train_scores['train_reg_FN'],                   
							train_scores['train_reg_TP'], 
							train_scores['train_clf_roc_auc'], 
							train_scores['train_clf_accuracy'],   
							train_scores['train_clf_precision'],
							train_scores['train_clf_recall'],    
							train_scores['train_clf_f1'],               
							train_scores['train_clf_FPR'],              
							train_scores['train_clf_TPR'],           
							train_scores['train_clf_TN'],             
							train_scores['train_clf_FP'],              
							train_scores['train_clf_FN'],                  
							train_scores['train_clf_TP']
						])

    with open('results_params_imps.csv', 'a') as f:
    	spamwriter = csv.writer(f)
    	spamwriter.writerow([
    						n,
    						exp_desc,
                            algo,
							lgb_params, 
                            params_clf,  
                            feature_importances
						])

################################################
# hyper param tuning
################################################
def get_best_params(train_df, type, t_col):

	def objective(params):
	    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
	    
	    # Keep track of evals
	    global ITERATION, lgb_params, params_clf
	    
	    ITERATION += 1
	    
	    # Retrieve the subsample if present otherwise set to 1.0
	    subsample = params['boosting_type'].get('subsample', 1.0)
	    
	    # Extract the boosting type
	    params['boosting_type'] = params['boosting_type']['boosting_type']
	    params['subsample'] = subsample
	    
	    # Make sure parameters that need to be integers are integers - subsample_freq
	    for parameter_name in ['num_leaves', 'min_child_weight', 'n_estimators']:
	        params[parameter_name] = int(params[parameter_name])
	    
	    start = timer()
	    
	    # set params
	    params_clf = params
	    lgb_params = params

	    # get traindf

	    # Perform n_folds cross validation
	    cv_results = train_XGB(train_df, type, t_col, default_param=False)
	    
	    run_time = timer() - start

	    # Dictionary with information for evaluation
	    return {'loss': cv_results, 'params': params, 'iteration': ITERATION, 
	            'train_time': run_time, 'status': STATUS_OK}

	#'subsample_freq': hp.quniform('subsample_freq', 0, 100, 5),
       
	# Define the search space
	space = {
	    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.1, 1)}, 
	                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.1, 1)},
	                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
	    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
	    'n_estimators': hp.quniform('n_estimators', 500, 2000, 100),
	    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
	    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
	    'min_child_weight': hp.quniform('min_child_weight', 20, 200, 5),
	    'min_split_gain': hp.uniform('min_split_gain', 0.00001, 1.0),
	    'reg_alpha': hp.uniform('reg_alpha', 0.0001, 1.0),
	    'reg_lambda': hp.uniform('reg_lambda', 0.0001, 1.0),
	    'colsample_bytree': hp.uniform('colsample_by_tree', 0.1, 1.0)
	}
	tpe_algorithm = tpe.suggest
	bayes_trials = Trials()
	global  ITERATION

	ITERATION = 0
	MAX_EVALS = 30

	# Run optimization
	best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

	# Sort the trials with lowest loss (highest AUC) first
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
	print(bayes_trials_results[:2])
	print(best)


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

def get_train_eval_split(n):
	skf = StratifiedKFold(n_splits=n)
	return skf

def train_XGB(train_df, type, t_col, default_param=True):
	splits = 3
	skf = get_train_eval_split(splits)

	lgb_params, params_clf, exp_desc, algo = get_my_config(default_param)


	cols = list(train_df.columns.values)
	cols.remove(t_col)
	cols.remove('fullVisitorId')
	#print(cols)
	X = train_df[cols]
	y = train_df[t_col]
	y_split = (train_df[t_col] > 0)*1
	#print(y_split.nunique())
	
	th = 95
	
	oof_clf_preds = np.zeros(X.shape[0])
	oof_clf_gt_th = np.zeros(X.shape[0])
	oof_reg_preds = np.zeros(X.shape[0])
	feature_importances = {'clf':[], 'reg':[]}
	print(lgb_params)

	for i, (train_index, val_index) in enumerate(skf.split(X, y_split)):

		model = lgb.LGBMRegressor(**lgb_params)
		model.fit(
                X.iloc[train_index], y.iloc[train_index],
                eval_set=[(X.iloc[val_index], y.iloc[val_index])],
                eval_metric='rmse',
                early_stopping_rounds=100,
                verbose=100
            )

		# predict train and val for getting scores
		y_pred = model.predict(X.iloc[val_index])
		thresh = np.percentile(y_pred, th)
		#print("th value", np.percentile(y_pred, th))
		oof_reg_preds[val_index] = y_pred

		y_pred[y_pred < thresh] = 0
		y_zero = np.zeros_like(y.iloc[val_index])
		y_true = y.iloc[val_index]

		y_train_true = y.iloc[train_index]
		y_train_pred = model.predict(X.iloc[train_index])
		y_train_pred[y_train_pred < thresh] = 0

		if(i==0):
			train_scores = get_scores(y_train_true, y_train_pred, y_train_pred, 'train', 'reg')
		else:
			train_scores = train_scores+get_scores(y_train_true, y_train_pred, y_train_pred, 'train', 'reg')

		print("confusion matrix", confusion_matrix(y_true > 0, y_pred > 0))
		print("all zero rmse", cal_rmse(y_true, y_zero))
		print("val rmse", cal_rmse(y_true, y_pred))
		
		# add feature importances
		feature_imp = pd.Series(index= X.columns.values, data= model.feature_importances_)
		#print(feature_imp.sort_values(ascending=False).head(7))
		#print(feature_imp[feature_imp < 0.002].index.values)
		#print(feature_imp.describe())
		feature_importances['reg'].append({i: feature_imp.sort_values(ascending=False).head(20)})

		"""
		if(default_param):
			with open("./models/20170801_"+str(i)+".pickle","wb") as f:
				pickle.dump(model, f)
		"""
		"""
		# train the classification model
		print(params_clf)
		
		model_clf = lgb.LGBMClassifier(**params_clf)
		model_clf.fit(
                X.iloc[train_index], y_split.iloc[train_index],
                eval_set=[(X.iloc[val_index], y_split.iloc[val_index])],
                eval_metric='AUC',
                early_stopping_rounds=100,
                verbose=100
            )

		# predict train and val for getting scores
		y_true = y_split.iloc[val_index]
		y_pred = model_clf.predict_proba(X.iloc[val_index])[:,1]
		oof_clf_preds[val_index] = y_pred
		thresh = np.percentile(y_pred, th)
		y_pred = (y_pred > thresh)*1
		oof_clf_gt_th[val_index] = y_pred

		y_train_true = y_split.iloc[train_index]
		y_train_predict_proba = model_clf.predict_proba(X.iloc[train_index])[:,1]
		y_train_pred = (y_train_predict_proba > thresh)*1
		
		if(i==0):
			train_scores_clf = get_scores(y_train_true, y_train_pred, y_train_predict_proba, 'train', 'clf')
		else:
			train_scores_clf = train_scores_clf+get_scores(y_train_true, y_train_pred, y_train_predict_proba, 'train', 'clf')


		#unique, counts = np.unique(y_true, return_counts=True)
		#print(np.asarray((unique, counts)).T)
		#unique, counts = np.unique(y_pred, return_counts=True)
		#print(np.asarray((unique, counts)).T)
		print("confusion matrix", confusion_matrix(y_true, y_pred))

		# add feature importance
		feature_imp = pd.Series(index= X.columns.values, data= model_clf.feature_importances_)
		print(feature_imp.sort_values(ascending=False).head(10))
		feature_importances['clf'].append({i: feature_imp.sort_values(ascending=False).head(20)})
		"""
	#results_df = pd.DataFrame({'actual': y[y>0], 'predicted': oof_reg_preds[y>0]})
	#print(results_df.describe())
	#train_scores = train_scores/3
	#train_scores_clf = train_scores_clf/3
	#train_scores = train_scores.append(train_scores_clf)
	#print(train_scores)
	#scores_df = get_scores(y_split, oof_clf_gt_th, oof_clf_preds, 'test', 'clf')	
	#scores_df = scores_df.append(get_scores(y, oof_reg_preds, oof_reg_preds, 'test', 'reg'))	
	#log_metrics(scores_df, train_scores, feature_importances)
	pd.DataFrame({'true_vals': y.values, 'predictions':oof_reg_preds}).to_csv('oof_predictions_20170501.csv', index=False)
	scores_df = get_scores(y, oof_reg_preds, oof_reg_preds, 'test', 'reg')
	print(scores_df)
	print("test reg RMSE - ", scores_df['test_reg_rmse'])
	return scores_df['test_reg_rmse']                  

def train_XGB_add_classifcation_as_reg_feature(train_df, type, t_col):
	splits = 3	
	skf = get_train_eval_split(splits)
	lgb_params, params_clf, exp_desc, algo = get_my_config()
	
	cols = list(train_df.columns.values)
	cols.remove(t_col)
	cols.remove('fullVisitorId')
	print(cols)
	X = train_df[cols]
	y = train_df[t_col]
	y_split = (train_df[t_col] > 0)*1
	print(y_split.nunique())
	th = 95

	oof_clf_preds = np.zeros(X.shape[0])
	oof_clf_gt_th = np.zeros(X.shape[0])
	feature_importances = {'clf':[], 'reg':[]}

	for i, (train_index, val_index) in enumerate(skf.split(X, y_split)):

		# train the classification model
		model_clf = lgb.LGBMClassifier(**params_clf)
		model_clf.fit(
                X.iloc[train_index], y_split.iloc[train_index],
                eval_set=[(X.iloc[val_index], y_split.iloc[val_index])],
                eval_metric='AUC',
                early_stopping_rounds=100,
                verbose=100
            )

		y_true = y_split.iloc[val_index]
		y_pred = model_clf.predict_proba(X.iloc[val_index])[:,1]
		oof_clf_preds[val_index] = y_pred
		thresh = np.percentile(y_pred, th)
		y_pred = (y_pred > thresh)*1
		oof_clf_gt_th[val_index] = y_pred

		y_train_true = y_split.iloc[train_index]
		y_train_predict_proba = model_clf.predict_proba(X.iloc[train_index])[:,1]
		y_train_pred = (y_train_predict_proba > thresh)*1
		
		if(i==0):
			train_scores_clf = get_scores(y_train_true, y_train_pred, y_train_predict_proba, 'train', 'clf')
		else:
			train_scores_clf = train_scores_clf+get_scores(y_train_true, y_train_pred, y_train_predict_proba, 'train', 'clf')

		# get feature importances
		print("confusion matrix", confusion_matrix(y_true, y_pred))
		feature_imp = pd.Series(index= X.columns.values, data=model_clf.feature_importances_)
		print(feature_imp.sort_values(ascending=False).head(10))
		feature_importances['clf'].append({i: feature_imp.sort_values(ascending=False).head(20)})
	
	X['non_zero_proba'] = oof_clf_preds
	print(X['non_zero_proba'].sum())

	oof_reg_preds = np.zeros(X.shape[0])
	for i, (train_index, val_index) in enumerate(skf.split(X, y_split)):
		model = lgb.LGBMRegressor(**lgb_params)
		use_col = list(X.columns)

		model.fit(
                X.iloc[train_index].loc[:, use_col], y.iloc[train_index],
                eval_set=[(X.iloc[val_index].loc[:, use_col], y.iloc[val_index])],
                eval_metric='rmse',
                early_stopping_rounds=100,
                verbose=100
            )

		y_pred = model.predict(X.iloc[val_index].loc[:, use_col])
		thresh = np.percentile(y_pred, th)
		print("th value", np.percentile(y_pred, th))

		y_pred[y_pred < thresh] = 0
		y_true = y.iloc[val_index]
		y_zero = np.zeros_like(y.iloc[val_index])

		oof_reg_preds[val_index] = y_pred
		
		y_train_true = y.iloc[train_index]
		y_train_pred = model.predict(X.iloc[train_index].loc[:, use_col])
		y_train_pred[y_train_pred < thresh] = 0

		if(i==0):
			train_scores = get_scores(y_train_true, y_train_pred, y_train_pred, 'train', 'reg')
		else:
			train_scores = train_scores+get_scores(y_train_true, y_train_pred, y_train_pred, 'train', 'reg')

		print("confusion matrix", confusion_matrix(y_true > 0, y_pred > 0))
		print("all zero rmse", cal_rmse(y_true, y_zero))
		print("val rmse", cal_rmse(y_true, y_pred))

		# for feature importances
		feature_imp = pd.Series(index= X.loc[:, use_col].columns.values, data= model.feature_importances_)
		print(feature_imp.sort_values(ascending=False).head(10))
		feature_importances['reg'].append({i: feature_imp.sort_values(ascending=False).head(20)})
	
	train_scores = train_scores/3
	train_scores_clf = train_scores_clf/3
	train_scores = train_scores.append(train_scores_clf)
	print(train_scores)
	scores_df = get_scores(y_split, oof_clf_gt_th, oof_clf_preds, 'test', 'clf')	
	scores_df = scores_df.append(get_scores(y, oof_reg_preds, oof_reg_preds, 'test', 'reg'))	
	log_metrics(scores_df, train_scores, feature_importances)

################################
# main function 
################################
@click.command()
@click.argument('input_filepath_x', type=click.Path(exists=True))
@click.argument('input_filepath_y', type=click.Path())
@click.argument('type')
def main(input_filepath_x, input_filepath_y, type):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features and target variable')
    x = pd.read_csv(input_filepath_x, dtype={'fullVisitorId': 'str'})
    y = pd.read_csv(input_filepath_y,dtype={'fullVisitorId': 'str'})

    x = preprocess_x(x)
    train = pd.merge(x, y, on='fullVisitorId', how='left')
    train = train.fillna(0)
    #get_best_params(train, 'reg', 'target')
    train_XGB(train, 'reg', 'target')
    #train_XGB_add_classifcation_as_reg_feature(train, 'reg', 'target')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()