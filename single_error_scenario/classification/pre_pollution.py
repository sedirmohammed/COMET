from pandas import read_csv
from json import load as load_json
from classification.utils.DatasetModifier import *
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from classification.utils.util import drop_table
import argparse
import os
from config.definitions import ROOT_DIR


def add_meta_information(df, pollution_level, seed, mode, params):
    df['polluter'] = params['polluter_class']
    df['pollution_level'] = pollution_level
    df['seed'] = seed
    df['train/test'] = mode
    return df


def main(error_type, ds_name, ds_path, metadata, database_url):
    database_engine = create_engine(database_url, echo=True, connect_args={'timeout': 1000})

    drop_table(f'pre_pollution_{ds_name}_{error_type.__name__}', database_engine)

    results_df = pd.DataFrame()
    for random_seed in metadata['random_seeds']:
        np.random.seed(random_seed)
        print(f'Current random seed: {random_seed}')
        df = read_csv(ds_path)

        for categorical_col in metadata[ds_name]['categorical_cols']:
            df[categorical_col] = df[categorical_col].astype(str)
        for numerical_col in metadata[ds_name]['numerical_cols']:
            df[numerical_col] = df[numerical_col].astype(float)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=1, stratify=df[metadata[ds_name]['target']])

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df['id'] = train_df.index
        test_df['id'] = test_df.index
        train_df_polluted = train_df.copy()
        test_df_polluted = test_df.copy()

        features_to_pre_pollute = list(df.columns)
        features_to_pre_pollute.remove(metadata[ds_name]['target'])

        for pollution_step in np.arange(0.0, metadata['pre_pollution_level'] + 0.01, metadata['pollution_level_step_size']):
            pollution_step = round(pollution_step, 2)
            print(f'Current pollution step: {pollution_step}')

            pollution_level_setting = dict.fromkeys(features_to_pre_pollute, pollution_step)

            train_df_polluted = error_type(pollution_level_setting, train_df, train_df_polluted, metadata[ds_name]).pollute()
            test_df_polluted = error_type(pollution_level_setting, test_df, test_df_polluted, metadata[ds_name]).pollute()

            train_df_polluted_temp = train_df_polluted.copy()
            train_df_polluted_temp['polluter'] = error_type.__name__
            train_df_polluted_temp['seed'] = random_seed
            train_df_polluted_temp['pollution_level'] = pollution_step
            train_df_polluted_temp['train/test'] = 'train'

            test_df_polluted_temp = test_df_polluted.copy()
            test_df_polluted_temp['polluter'] = error_type.__name__
            test_df_polluted_temp['seed'] = random_seed
            test_df_polluted_temp['pollution_level'] = pollution_step
            test_df_polluted_temp['train/test'] = 'test'

            results_df = pd.concat([results_df, train_df_polluted_temp], ignore_index=True)
            results_df = pd.concat([results_df, test_df_polluted_temp], ignore_index=True)

    print(results_df.head(25).to_string())
    results_df.to_sql(name=f'pre_pollution_{ds_name}_{error_type.__name__}', con=database_engine, if_exists='append', index=False)


if __name__ == "__main__":

    error_types = {'MissingValuesModifier': MissingValuesModifier,
                   'CategoricalShiftModifier': CategoricalShiftModifier,
                   'ScalingModifier': ScalingModifier,
                   'GaussianNoiseModifier': GaussianNoiseModifier}

    parser = argparse.ArgumentParser()
    parser.add_argument('--error_type', default='MissingValuesModifier', type=str, help='Set the error type to use for the experiment.')

    parser.add_argument('--overwrite', default='n', type=str, help='Set if existing pre-pollution will get overwritten.')

    ds_path = os.path.join(ROOT_DIR, 'data', 'SouthGermanCredit.csv')
    parser.add_argument('--dataset', default=ds_path, type=str, help='Set the path to the dataset to use for the experiment.')

    metadata_path = os.path.join(ROOT_DIR, 'metadata.json')
    parser.add_argument('--metadata', default=metadata_path, type=str, help='Set the path to metadata.json file to use for the experiment.')

    database_url = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
    parser.add_argument('--database_url', default=database_url, type=str, help='Set the url for the database to use for the experiment.')

    args = parser.parse_args()

    error_type_str = args.error_type
    error_type = error_types[error_type_str]

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    ds_name = args.dataset
    ds_name = ds_name.split('/')[-1]

    if args.overwrite != 'y':
        print('You chose not to overwrite the existing pre-polluted versions. Quitting...')
        quit()
    main(error_type, ds_name, args.dataset, metadata, args.database_url)
