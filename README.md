# Readme

## Environment setup
* Create a virtual environment with python 3.9
* Install requirements.txt

## Metadata
* Specify per dataset:
  - dataset name
  - name of categorical columns (categorical_cols)
  - name of numerical columns (numerical_cols)
  - placeholders for missing values (placeholders)
  - number of samples (n_samples)
  - name of the target column (target)

    
## Pre-pollution based datasets
* Run ```generate_pre_pollution_settings.py```, specify dataset path, metadata name and database url
* Run ```pre_pollution.py```, specify error type, dataset path, metadata name and database url

## CleanML based datasets
* Run ```generate_cleanml_settings.py```, specify dataset path, metadata name and database url

## Hyperparameter search
* Run ```find_hyper_params.py``` (check settings in the script)

## Run experiments
* Run the script for the respective method: 
  - COMET: ```dynamic_greedy.py```
  - FIR: ```static_feature_importance_greedy.py``` 
  - RR: ```completely_random_recommendations.py```
  - Oracle: ```dynamic_greedy_optimal.py```
  
## Experiment results
[https://my.hidrive.com/lnk/k9imgg1p](https://my.hidrive.com/lnk/k9imgg1p)
Password: *\_COMET\_*



