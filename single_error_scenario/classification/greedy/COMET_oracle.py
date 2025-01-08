import time
from sqlalchemy import create_engine
from classification.utils.DatasetModifier import *
from classification.utils.artifical_pollution import *
from classification.experiments import *
from json import load as load_json
from classification.utils.util import load_pre_pollution_df, get_pre_pollution_settings, delete_entries_from_table
from util import start_logging
import argparse
import os
from config.definitions import ROOT_DIR
from decimal import Decimal
pd.options.mode.chained_assignment = None


def select_cleaning_setting(polluted_dfs, feature_wise_pollution_level, budget):

    cleaning_setting = {'feature': None, 'pollution_level': -1, 'predicted_poly_reg_f1': -1, 'real_f1': -1, 'used_budget': -1, 'f1_gain_predicted': -1}
    for feature_entry in polluted_dfs:
        current_polluted_df = feature_entry['polluted_df'].copy()
        current_polluted_df = current_polluted_df[current_polluted_df['pollution_level'] == current_polluted_df['pollution_level'].min()]

        current_polluted_df['predicted_poly_reg_f1'] = current_polluted_df['predicted_poly_reg_f1'].astype(float)
        current_polluted_df['used_budget'] = 1
        current_polluted_df = current_polluted_df[current_polluted_df['used_budget'] <= budget]
        current_polluted_df = current_polluted_df.tail(1)

        if current_polluted_df.empty:
            print('current_polluted_df is empty')
            continue

        current_polluted_df = current_polluted_df.sort_values(by=['real_f1'], ascending=False)
        current_polluted_df = current_polluted_df.head(1)

        if cleaning_setting['real_f1'] < current_polluted_df['real_f1'].values[0]:
            cleaning_setting['feature'] = feature_entry['feature']
            cleaning_setting['pollution_level'] = current_polluted_df['pollution_level'].values[0]
            cleaning_setting['predicted_poly_reg_f1'] = -1
            cleaning_setting['real_f1'] = current_polluted_df['real_f1'].values[0]
            cleaning_setting['used_budget'] = round(current_polluted_df['used_budget'].values[0], 0)
            cleaning_setting['f1_gain_predicted'] = -1

    return cleaning_setting


def update_feature_wise_pollution_level(feature_wise_pollution_level, cleaning_setting):
    print('before,', feature_wise_pollution_level)

    feature = cleaning_setting['feature']

    # Update and round for 'train'
    if feature_wise_pollution_level['train'][feature] > 0:
        train_val = Decimal(str(feature_wise_pollution_level['train'][feature])) - Decimal('0.01')
        feature_wise_pollution_level['train'][feature] = float(train_val.quantize(Decimal('0.01')))

    # Update and round for 'test'
    if feature_wise_pollution_level['test'][feature] > 0:
        test_val = Decimal(str(feature_wise_pollution_level['test'][feature])) - Decimal('0.01')
        feature_wise_pollution_level['test'][feature] = float(test_val.quantize(Decimal('0.01')))

    print('after,', feature_wise_pollution_level)
    return feature_wise_pollution_level


def write_cleaning_setting_to_db(cleaning_setting, iteration, ds_name, experiment_name, mod_name, original_f1_score, original_budget, pre_pollution_setting_id, database_engine):
    current_cleaning_setting = cleaning_setting.copy()
    current_cleaning_setting['iteration'] = iteration
    current_cleaning_setting['dataset'] = ds_name
    current_cleaning_setting['experiment'] = experiment_name
    current_cleaning_setting['polluter'] = mod_name
    current_cleaning_setting['original_f1_score'] = original_f1_score
    current_cleaning_setting['original_budget'] = original_budget
    current_cleaning_setting['feature'] = current_cleaning_setting['feature']
    current_cleaning_setting['pre_pollution_setting_id'] = pre_pollution_setting_id

    current_cleaning_setting = pd.DataFrame(current_cleaning_setting, index=[0])
    table_name = f'cleaning_schedule_optimal_{ds_name}_{experiment_name}_{mod_name}'
    #current_cleaning_setting.to_sql(table_name, con=database_engine, if_exists='append', index=False)
    with open(f'{ROOT_DIR}/slurm/oracle/RESULTS/{table_name}_{pre_pollution_setting_id}.csv', 'a') as f:
        if os.stat(f'{ROOT_DIR}/slurm/oracle/RESULTS/{table_name}_{pre_pollution_setting_id}.csv').st_size == 0:
            current_cleaning_setting.to_csv(f, header=True, index=False)
        else:
            current_cleaning_setting.to_csv(f, header=False, index=False)


def main(ml_algorithm, error_type, ds_name, original_budget, metadata, database_engine, pre_pollution_setting_ids):

    table_name = f'cleaning_schedule_optimal_{ds_name}_{ml_algorithm.__name__}_{error_type.__name__}'
    #delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids)

    pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=pre_pollution_setting_ids)
    print(pre_pollution_settings)

    for pollution_setting in pre_pollution_settings:
        start_time = time.time()

        cleaning_schedule = []
        pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
        if not os.path.exists(f'{ROOT_DIR}/slurm/oracle/RESULTS/'):
            os.makedirs(f'{ROOT_DIR}/slurm/oracle/RESULTS/')
        else:
            if os.path.exists(f'{ROOT_DIR}/slurm/oracle/RESULTS/{table_name}_{pre_pollution_setting_id}.csv'):
                os.remove(f'{ROOT_DIR}/slurm/oracle/RESULTS/{table_name}_{pre_pollution_setting_id}.csv')

        iteration = 1
        BUDGET = 50
        while BUDGET > 0:
            print(f'Current config: {ml_algorithm.__name__}, {error_type.__name__}, {ds_name}, iteration {iteration}')

            ap = ArtificialPollution(metadata, str(database_engine.url), pollution_setting['pre_pollution_setting_id'], error_type)
            polluted_dfs, original_f1_score = ap.artificial_pollution(pre_pollution_df, ds_name, ml_algorithm, pollution_setting, iteration, skip_pollution=True)

            if len(polluted_dfs) == 0:
                print('Nothing to clean anymore.')
                break

            cleaning_setting = select_cleaning_setting(polluted_dfs, pollution_setting, BUDGET)
            if cleaning_setting['feature'] is None:
                print('Nothing to clean anymore.')
                break
            write_cleaning_setting_to_db(cleaning_setting, iteration, ds_name, ml_algorithm.__name__, error_type.__name__, original_f1_score, original_budget, pre_pollution_setting_id, database_engine)

            cleaning_schedule.append(cleaning_setting)
            print(f'iteration {iteration}; cleaning_setting', cleaning_setting)

            pollution_setting = update_feature_wise_pollution_level(pollution_setting, cleaning_setting)
            print('feature_wise_pollution_level', pollution_setting)

            BUDGET = BUDGET - cleaning_setting['used_budget']
            print('Available budget:', BUDGET)

            iteration += 1
            print('cleaning_schedule', cleaning_schedule)
            print('Needed time for current pre-pollution setting', (time.time() - start_time), 'seconds')

        print('cleaning_schedule', cleaning_schedule)
        print('Needed time for all pre-pollution settings', (time.time() - start_time), 'seconds')


def get_cleaned_df(history_df, cleaning_setting, mode, metadata, ds_name, random_seed):
    cleaned_df = history_df.copy()

    cleaned_df = cleaned_df[cleaned_df['pollution_level'] == metadata['pre_pollution_level']]
    cleaned_df = cleaned_df[cleaned_df['train/test'] == mode]
    cleaned_df = cleaned_df[cleaned_df['seed'] == random_seed]
    cleaned_df = cleaned_df.set_index('id')

    cleaned_df = cleaned_df[metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols'] + [metadata[ds_name]['target']]]

    for setting in cleaning_setting:
        temp_df = history_df.copy()
        temp_df = temp_df[temp_df['train/test'] == mode]
        temp_df = temp_df[temp_df['pollution_level'] == setting['pollution_level']]
        temp_df = temp_df[temp_df['seed'] == random_seed]
        temp_df = temp_df[['id', setting['feature']]]
        temp_df = temp_df.set_index('id')

        cleaned_df = cleaned_df.join(temp_df, on='id', how='left', lsuffix='_old')
        cleaned_df = cleaned_df.drop(columns=[f'{setting["feature"]}_old'])
    cleaned_df = cleaned_df[metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols'] + [metadata[ds_name]['target']]]
    return cleaned_df


def filter_features_on_type(feature_candidates_for_cleaning, modifier, metadata, ds_name):
    feature_candidates_for_cleaning_temp = []
    for feature in feature_candidates_for_cleaning:
        if feature in metadata[ds_name]['categorical_cols']:
            feature_type = 'categorical_col'
        else:
            feature_type = 'numerical_col'
        if feature_type == modifier.restricted_to or modifier.restricted_to == '':
            feature_candidates_for_cleaning_temp.append(feature)
    return feature_candidates_for_cleaning_temp


if __name__ == "__main__":
    start_logging(cmd_out=True)

    ml_algorithms = {'SupportVectorMachineExperiment': SupportVectorMachineExperiment,
                     'MultilayerPerceptronExperiment': MultilayerPerceptronExperiment,
                     'KNeighborsExperiment': KNeighborsExperiment,
                     'GradientBoostingExperiment': GradientBoostingExperiment,
                     'RandomForrestExperiment': RandomForrestExperiment}

    error_types = {'MissingValuesModifier': MissingValuesModifier,
                   'CategoricalShiftModifier': CategoricalShiftModifier,
                   'ScalingModifier': ScalingModifier,
                   'GaussianNoiseModifier': GaussianNoiseModifier}

    parser = argparse.ArgumentParser()
    parser.add_argument('--ml_algorithm', default='SupportVectorMachineExperiment', type=str, help='Set the ml algorithm to use for the experiment.')
    parser.add_argument('--error_type', default='MissingValuesModifier', type=str, help='Set the error type to use for the experiment.')
    parser.add_argument('--dataset', default='SouthGermanCredit.csv', type=str, help='Set the dataset to use for the experiment.')
    parser.add_argument('--budget', default=1000, type=int, help='Set the available budget for the experiment.')
    metadata_path = os.path.join(ROOT_DIR, 'metadata.json')
    parser.add_argument('--metadata', default=metadata_path, type=str, help='Set the path to metadata.json file to use for the experiment.')

    database_url = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
    parser.add_argument('--database_url', default=database_url, type=str, help='Set the url for the database to use for the experiment.')
    parser.add_argument('--pre_pollution_settings', default='1 2 3', type=str, help='Set the pre pollution setting id s.')
    args = parser.parse_args()

    ml_algorithm_str = args.ml_algorithm
    chosen_ml_algorithm = ml_algorithms[ml_algorithm_str]

    error_type_str = args.error_type
    chosen_error_type = error_types[error_type_str]

    ds_name = args.dataset
    budget = args.budget
    pre_pollution_setting_ids = [int(i) for i in args.pre_pollution_settings.split(' ')]

    database_engine = create_engine(database_url, echo=True, connect_args={'timeout': 1000})

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    main(chosen_ml_algorithm, chosen_error_type, ds_name, budget, metadata, database_engine, pre_pollution_setting_ids)
