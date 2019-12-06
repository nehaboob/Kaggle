.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = googleanalytics
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pip install -U pip setuptools wheel
	pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
json_preprocess:
	python src/data/make_dataset.py ./data/raw/train_v2.csv ./data/processed/train_v2_flat.csv None None
	python src/data/make_dataset.py ./data/raw/test_v2.csv ./data/processed/test_v2_flat.csv None None

get_features:
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20160801_v3.csv None 20160801 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20160901_v3.csv None 20160901 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20161001_v3.csv None 20161001 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20161101_v3.csv None 20161101 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20161201_v3.csv None 20161201 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170101_v3.csv None 20170101 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170201_v3.csv None 20170201 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170301_v3.csv None 20170301 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170401_v3.csv None 20170401 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170501_v3.csv None 20170501 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170601_v3.csv None 20170601 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170701_v3.csv None 20170701 
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_feaures_20170801_v3.csv None 20170801 

get_test_features:
	python src/features/build_features.py ./data/processed/test_v2_flat.csv ./data/processed/user_feaures_test_20180501_v3.csv None 20180501 

get_predictions:
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20160801_v3.csv totals_transactions 20160801
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20160901_v3.csv totals_transactions 20160901
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20161001_v3.csv totals_transactions 20161001
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20161101_v3.csv totals_transactions 20161101
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20161201_v3.csv totals_transactions 20161201
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170101_v3.csv totals_transactions 20170101
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170201_v3.csv totals_transactions 20170201
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170301_v3.csv totals_transactions 20170301
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170401_v3.csv totals_transactions 20170401
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170501_v3.csv totals_transactions 20170501
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170601_v3.csv totals_transactions 20170601
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170701_v3.csv totals_transactions 20170701
	python src/features/build_features.py ./data/processed/train_v2_flat.csv ./data/processed/user_predictions_20170801_v3.csv totals_transactions 20170801

train_model:
	#python src/models/train_model.py ./data/processed/user_feaures_20170501.csv ./data/processed/user_predictions_20170501.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170501_v2.csv ./data/processed/user_predictions_20170501_v2.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20160801_v3.csv ./data/processed/user_predictions_20160801_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20160901_v3.csv ./data/processed/user_predictions_20160901_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20161001_v3.csv ./data/processed/user_predictions_20161001_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20161101_v3.csv ./data/processed/user_predictions_20161101_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20161201_v3.csv ./data/processed/user_predictions_20161201_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170101_v3.csv ./data/processed/user_predictions_20170101_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170201_v3.csv ./data/processed/user_predictions_20170201_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170301_v3.csv ./data/processed/user_predictions_20170301_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170401_v3.csv ./data/processed/user_predictions_20170401_v3.csv reg
	python src/models/train_model.py ./data/processed/user_feaures_20170501_v3.csv ./data/processed/user_predictions_20170501_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170601_v3.csv ./data/processed/user_predictions_20170601_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170701_v3.csv ./data/processed/user_predictions_20170701_v3.csv reg
	#python src/models/train_model.py ./data/processed/user_feaures_20170801_v3.csv ./data/processed/user_predictions_20170801_v3.csv reg

do_predictions:
	python src/models/predict_model.py ./models/20160801_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20160801.csv
	python src/models/predict_model.py ./models/20160801_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20160801.csv
	python src/models/predict_model.py ./models/20160801_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20160801.csv
	python src/models/predict_model.py ./models/20160901_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20160901.csv
	python src/models/predict_model.py ./models/20160901_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20160901.csv
	python src/models/predict_model.py ./models/20160901_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20160901.csv
	python src/models/predict_model.py ./models/20161001_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20161001.csv
	python src/models/predict_model.py ./models/20161001_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20161001.csv
	python src/models/predict_model.py ./models/20161001_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20161001.csv
	python src/models/predict_model.py ./models/20161101_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20161101.csv
	python src/models/predict_model.py ./models/20161101_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20161101.csv
	python src/models/predict_model.py ./models/20161101_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20161101.csv
	python src/models/predict_model.py ./models/20161201_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20161201.csv
	python src/models/predict_model.py ./models/20161201_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20161201.csv
	python src/models/predict_model.py ./models/20161201_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20161201.csv
	python src/models/predict_model.py ./models/20170101_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170101.csv
	python src/models/predict_model.py ./models/20170101_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170101.csv
	python src/models/predict_model.py ./models/20170101_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170101.csv
	python src/models/predict_model.py ./models/20170201_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170201.csv
	python src/models/predict_model.py ./models/20170201_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170201.csv
	python src/models/predict_model.py ./models/20170201_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170201.csv
	python src/models/predict_model.py ./models/20170301_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170301.csv
	python src/models/predict_model.py ./models/20170301_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170301.csv
	python src/models/predict_model.py ./models/20170301_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170301.csv
	python src/models/predict_model.py ./models/20170401_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170401.csv
	python src/models/predict_model.py ./models/20170401_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170401.csv
	python src/models/predict_model.py ./models/20170401_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170401.csv
	python src/models/predict_model.py ./models/20170501_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170501.csv
	python src/models/predict_model.py ./models/20170501_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170501.csv
	python src/models/predict_model.py ./models/20170501_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170501.csv
	python src/models/predict_model.py ./models/20170601_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170601.csv
	python src/models/predict_model.py ./models/20170601_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170601.csv
	python src/models/predict_model.py ./models/20170601_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170601.csv
	python src/models/predict_model.py ./models/20170701_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170701.csv
	python src/models/predict_model.py ./models/20170701_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170701.csv
	python src/models/predict_model.py ./models/20170701_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170701.csv
	python src/models/predict_model.py ./models/20170801_0.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_0_20170801.csv
	python src/models/predict_model.py ./models/20170801_1.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_1_20170801.csv
	python src/models/predict_model.py ./models/20170801_2.pickle ./data/processed/user_feaures_test_20180501_v3.csv ./reports/submission_2_20170801.csv

final_submission:
	python src/models/final_submit.py 
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
