from sqlalchemy import create_engine
from json import load as load_json
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import os
from config.definitions import ROOT_DIR
pd.options.mode.chained_assignment = None


def main(ds_name, metadata, database_url):
    database_engine = create_engine(database_url, echo=True)

    global_feature_wise_pollution_level = {}
    pre_pollution_settings = []

    for i in range(1, 11):
        pre_pollution_setting = get_pre_pollution_setting(ds_name, metadata)
        pre_pollution_setting['pre_pollution_setting_id'] = i
        pre_pollution_settings.append(pre_pollution_setting)
        for feature in pre_pollution_setting.keys():
            if feature not in global_feature_wise_pollution_level.keys():
                global_feature_wise_pollution_level[feature] = []
            global_feature_wise_pollution_level[feature].append(pre_pollution_setting[feature])

    for feature in global_feature_wise_pollution_level.keys():
        mean = round(np.mean(global_feature_wise_pollution_level[feature]), 4)
        variance = round(np.var(global_feature_wise_pollution_level[feature]), 4)
        print(f'{feature}, mean: {mean}, variance: {variance}')

    print(global_feature_wise_pollution_level)
    print(pre_pollution_settings)
    pre_pollution_settings_df = pd.DataFrame(pre_pollution_settings)
    pre_pollution_settings_df = pre_pollution_settings_df.round(2)
    print(pre_pollution_settings_df.to_string())
    table_name = f'pre_pollution_settings_{ds_name}'
    pre_pollution_settings_df.to_sql(table_name, con=database_engine, if_exists='replace', index=False)


def exponential_function(x, exp_lambda):
    return exp_lambda * math.exp(-exp_lambda * x)


def get_pre_pollution_setting(ds_name, metadata):
    lambda_param = 0.5
    def generate_pollution_value():
        random_x = np.random.exponential(1 / lambda_param)
        random_pollution_level = random_x * np.exp(-random_x / lambda_param)
        if random_pollution_level > 0.2:
            random_pollution_level = 0.2
        return random_pollution_level

    pre_pollution_setting = {}

    # Continue until the sum of pre-pollution settings across all features is >= 0.5
    while True:
        for feature in metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols']:
            pre_pollution_setting[feature] = generate_pollution_value()
        pre_pollution_setting[metadata[ds_name]['target']] = 0.0

        # Check if the sum of pre-pollution settings is >= 0.5
        if sum(pre_pollution_setting.values()) >= 0.5:
            break

    print(pre_pollution_setting)

    pre_pollution_setting = dict(sorted(pre_pollution_setting.items(), key=lambda item: item[1], reverse=True))
    plt.bar(pre_pollution_setting.keys(), pre_pollution_setting.values())
    plt.xticks(rotation=90)
    plt.ylim(0, 0.2)
    plt.title(f'Feature-wise pollution level with lambda = {lambda_param}')
    #plt.show()

    return pre_pollution_setting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='SouthGermanCredit.csv', type=str, help='Set the dataset to use for the experiment.')

    metadata_path = os.path.join(ROOT_DIR, 'metadata.json')
    parser.add_argument('--metadata', default=metadata_path, type=str, help='Set the path to metadata.json file to use for the experiment.')

    database_url = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
    parser.add_argument('--database_url', default=database_url, type=str, help='Set the url for the database to use for the experiment.')
    args = parser.parse_args()

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    ds_name = args.dataset

    print(f'You are about to generate pre-pollution settings for the {ds_name} dataset. '
          f'This will overwrite the existing pre-pollution settings for this dataset')
    proceed = input('Do you want to continue? (y/n): ')
    if proceed != 'y':
        quit()
    main(ds_name, metadata, args.database_url)
