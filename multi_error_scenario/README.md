# Readme

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

## CleanML based datasets
* Run ```generate_cleanml_settings.py```, specify dataset path, metadata name and database url

## Hyperparameter search
* Run ```find_hyper_params_multi_error.py``` (check settings in the script)

## Run experiments
* Run the script for the respective method: 
  - COMET: ```COMET.py```
  - FIR: ```FIR.py``` 
  - RR: ```RR.py```
  - CL: ```CL.py```
  - Oracle: ```COMET_oracle.py```
  - ActiveClean: ```AC.py```

## Streamlit
Start the streamlit app from the multi-error scenario directory: ```python -m streamlit run classification/streamlit/main_page.py```

