from sklearn.model_selection import StratifiedKFold, KFold
import csv
import pandas as pd
import pickle
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def spilt_dataset(ids, folds):
	skf = KFold(n_splits=folds)
	for i, (train_index, val_index) in enumerate(skf.split(ids)):
		with open('./data/interim/train_'+str(i)+'.pickle', 'wb') as f:
			pickle.dump(train_index, f)

		with open('./data/interim/val_'+str(i)+'.pickle', 'wb') as f:
			pickle.dump(val_index, f)

def spilt_dataset_skf(df, folds):
	skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=15)
	for i, (train_index, val_index) in enumerate(skf.split(df['card_id'], df['class'])):
		with open('./data/interim/train_skf_'+str(i)+'.pickle', 'wb') as f:
			pickle.dump(train_index, f)

		with open('./data/interim/val_skf_'+str(i)+'.pickle', 'wb') as f:
			pickle.dump(val_index, f)

@click.command()
def main():
	""" Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
	logger = logging.getLogger(__name__)
	logger.info('making final data set from raw data')
	train=pd.read_csv('./data/raw/train.csv')
	#spilt_dataset(train['card_id'], 5)
	train['class'] = 0
	train.loc[train.target < -30, 'class'] = 1
	spilt_dataset_skf(train[['card_id', 'class']], 5)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()