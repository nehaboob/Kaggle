import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import random
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
from sklearn.metrics import r2_score

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
def select_features(X, y, params, base_features):
	final_features=base_features.copy()
	feature_list = list(X.columns)
	feature_list = [feature for feature in feature_list if feature not in base_features]
	random.shuffle(feature_list)
	exp_id = randint(0, 10000)
	X_interim = X[final_features]
	cv_best = train_XGB(X_interim, y, params, 5, False, exp_id)

	for feature in feature_list:
		print(feature)
		interim_features = final_features+[feature]
		exp_id = randint(0, 10000)
		X_interim = X[interim_features]
		cv_interim = train_XGB(X_interim, y, params, 5, False, exp_id)
		if cv_interim > cv_best:
			final_features.append(feature)
			cv_best = cv_interim
			print("added to final_features")
		#print("interim_features", cv_interim, interim_features)
	print("final_features", final_features)
	return final_features

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
                categorical_feature=['feature_1', 'feature_2', 'feature_3'],
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
		feature_imp = pd.Series(index= X.columns.values, data= model.feature_importances_)
		log_metrics(val_score, tmp_train_score, feature_imp.sort_values(ascending=False).head(20), params, i, exp_id)
		feature_imp.sort_values(ascending=False).to_csv('features.csv')
		if(test_results):
			test_pred = test_pred+(model.predict(X_test)/5)
	
	if(test_results):
		pd.DataFrame({'test_pred': test_pred}).to_csv('./data/interim/test_pred.csv', index=False)
	
	#store oof predictions
	#pd.DataFrame({'true_vals': y.values, 'predictions':oof_reg_preds}).to_csv('oof_predictions_20170501.csv', index=False)
	scores_df = get_scores(y, oof_reg_preds,'test')
	r2 = r2_score(y, oof_reg_preds)
	n=X.shape[0]
	p=X.shape[1]
	adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

	print("test RMSE - ", scores_df['test_rmse'], 'test adj r2 - ', adj_r2, "train RMSE - ", train_scores['train_rmse']/splits)

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

	#return scores_df['test_rmse']                  
	return adj_r2

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
    cols = [col for col in train.columns if col not in ['first_active_month', 'card_id', 'target', 'Unnamed: 0']]

    X=train[cols]
    y=train['target']

    params = {'boosting_type': 'gbdt', 'colsample_bytree': 0.5170233956444398, 'learning_rate': 0.01143545759270505, 'min_child_weight': 200, 'min_split_gain': 0.24720655947734432, 'n_estimators': 8700, 'num_leaves': 31, 'reg_alpha': 0.14446986536955275, 'reg_lambda': 0.8821246705659376, 'subsample': 0.9444712859527173, 'subsample_for_bin': 140000.0}
    base_features = ['feature_1', 'feature_2', 'feature_3', 'hist_merch_tran_ratio']
    final_features = select_features(X, y, params, base_features)
    print(final_features)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()