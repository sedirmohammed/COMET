import math
import pandas as pd
import numpy as np
import argparse
import os
from json import load as load_json
from sqlalchemy import create_engine

from classification.utils.util import drop_table
from config.definitions import ROOT_DIR


def calculate_pollution_levels(dirty_data, clean_data):
    pollution_levels = {}
    for feature in dirty_data.columns:
        # Count the number of different (dirty) entries for the feature
        dirty_count = np.sum(dirty_data[feature] != clean_data[feature])
        # Calculate the pollution level
        pollution_level = dirty_count / len(dirty_data)
        if 0 < pollution_level < 0.01:
            pollution_level = 0.01
        pollution_levels[feature] = pollution_level
    return pollution_levels


def clean_feature_by_dirty_percentage(data, feature, clean_data):
    # Identify dirty entries for the feature
    dirty_entries = data[feature] != clean_data[feature]

    entries_to_clean = int(math.ceil(len(data) * 0.01))

    # Select random indices of dirty entries for cleaning
    dirty_indices = data[dirty_entries].index

    # Ensuring entries_to_clean does not exceed the number of dirty_indices
    entries_to_clean = min(entries_to_clean, len(dirty_indices))

    np.random.seed(277712)

    indices_to_clean = np.random.choice(dirty_indices, size=entries_to_clean, replace=False)

    print('entries_to_clean', entries_to_clean)
    print('indices_to_clean', indices_to_clean)

    # Replace selected indices with values from the clean dataset
    data.loc[indices_to_clean, feature] = clean_data.loc[indices_to_clean, feature]
    return data


def construct_cleaning_dataset(dirty_data, clean_data, data_type='train'):
    # Initialize pollution levels
    pollution_levels = calculate_pollution_levels(dirty_data, clean_data)
    pollution_levels = {feature: round(pollution_levels[feature], 2) for feature in pollution_levels}

    max_pollution_level = max(pollution_levels.values())
    dirty_data['pollution_level'] = max_pollution_level
    results_df = dirty_data.copy()

    # Iteratively clean the dataset
    while max(pollution_levels.values()) > min(pollution_levels.values()):
        max_pollution_level = max(pollution_levels.values())
        features_to_clean = [feature for feature, level in pollution_levels.items() if level == max_pollution_level]

        for feature in features_to_clean:
            dirty_data = clean_feature_by_dirty_percentage(dirty_data, feature, clean_data)
            pollution_levels[feature] -= 0.01
            # round pollution level to 2 decimal places
            pollution_levels[feature] = round(pollution_levels[feature], 2)


        pollution_levels = {feature: round(pollution_levels[feature], 2) for feature in pollution_levels}
        dirty_data['pollution_level'] = max_pollution_level - 0.01
        dirty_data['pollution_level'] = dirty_data['pollution_level'].apply(lambda x: round(x, 2))

        results_df = pd.concat([results_df, dirty_data.copy()], ignore_index=True)

    results_df['train/test'] = data_type
    results_df['pollution_level'] = results_df['pollution_level'].apply(lambda x: round(x, 2))
    return results_df


def calculate_and_store_pre_pollution_settings(dirty_data, clean_data, ds_name, data_type='train'):
    pollution_levels = calculate_pollution_levels(dirty_data, clean_data)
    pollution_levels = {feature: round(pollution_levels[feature], 2) for feature in pollution_levels}
    print(f'pollution_levels: {pollution_levels}')
    table_name = f'pre_pollution_settings_{ds_name}_{data_type}'
    # pollution level to df
    pre_pollution_settings_df = pd.DataFrame([pollution_levels])
    pre_pollution_settings_df['pre_pollution_setting_id'] = 1
    pre_pollution_settings_df.to_sql(table_name, con=database_engine, if_exists='replace', index=False)


def delete_nan_values(df, metadata, ds_name):
    for col in metadata[ds_name]['numerical_cols']:
        df[col] = df[col].apply(lambda x: -1 if pd.isnull(x) else x)
    # if a categorical column contains nan values, convert them to 'nan'
    for col in metadata[ds_name]['categorical_cols']:
        df[col] = df[col].apply(lambda x: 'nan' if pd.isnull(x) else x)
    return df


def main(ds_name, error_type, database_engine):

    if ds_name == 'Credit' or ds_name == 'Airbnb' or ds_name == 'Titanic':
        dirty_data_train = pd.read_csv(f'{ROOT_DIR}/data/{ds_name}/{error_type}/dirty_train.csv')
        clean_data_train = pd.read_csv(f'{ROOT_DIR}/data/{ds_name}/{error_type}/clean_HC_impute_holoclean_train.csv')
        dirty_data_test = pd.read_csv(f'{ROOT_DIR}/data/{ds_name}/{error_type}/dirty_test.csv')
        clean_data_test = pd.read_csv(f'{ROOT_DIR}/data/{ds_name}/{error_type}/clean_HC_impute_holoclean_test.csv')

        dirty_data_train = delete_nan_values(dirty_data_train, metadata, ds_name)
        clean_data_train = delete_nan_values(clean_data_train, metadata, ds_name)
        dirty_data_test = delete_nan_values(dirty_data_test, metadata, ds_name)
        clean_data_test = delete_nan_values(clean_data_test, metadata, ds_name)
    else:
        print('Not routine for dataset found')
        quit()


    calculate_and_store_pre_pollution_settings(dirty_data_train, clean_data_train, ds_name, data_type='train')
    calculate_and_store_pre_pollution_settings(dirty_data_test, clean_data_test, ds_name, data_type='test')

    results_df = pd.DataFrame()
    temp_df = construct_cleaning_dataset(dirty_data_train, clean_data_train, data_type='train')
    results_df = pd.concat([results_df, temp_df], ignore_index=True)
    temp_df = construct_cleaning_dataset(dirty_data_test, clean_data_test, data_type='test')
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

    error_type_map = {'missing-values': 'MissingValuesModifier', 'outliers': 'ScalingModifier'}
    results_df['seed'] = 277712
    results_df['polluter'] = error_type_map[error_type]

    table_name = f'pre_pollution_{ds_name}_{error_type_map[error_type]}'
    drop_table(table_name, database_engine)
    results_df.to_sql(name=table_name, con=database_engine, if_exists='append', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--error_type', default='missing-values', type=str, help='Set the error type to use for the experiment.')
    parser.add_argument('--dataset', default='Titanic', type=str, help='Set the dataset to use for the experiment.')
    metadata_path = os.path.join(ROOT_DIR, 'metadata.json')
    parser.add_argument('--metadata', default=metadata_path, type=str, help='Set the path to metadata.json file to use for the experiment.')

    args = parser.parse_args()

    dataset = args.dataset
    error_type = args.error_type

    database_url = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
    database_engine = create_engine(database_url, echo=True, connect_args={'timeout': 1000})

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    main(dataset, error_type, database_engine)
