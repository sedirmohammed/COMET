import time
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.exc import NoSuchTableError
from classification.utils.DatasetModifier import *
from classification.utils.artifical_pollution import *
from classification.experiments import *
from json import load as load_json
from classification.utils.util import write_time_measurement_to_db, load_pre_pollution_df, get_pre_pollution_settings, delete_entries_from_table
import warnings
import argparse
import os
from classification.utils.strategies.cleaning_strategy import CleaningSetting1, CleaningSetting2, CleaningSetting3, CleaningSetting4, CleaningSetting5
from classification.utils.strategies.data_cleaning import DataCleaning
from config.definitions import ROOT_DIR
from decimal import Decimal
pd.options.mode.chained_assignment = None


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

    metadata = MetaData(bind=database_engine)
    table_name = f'cleaning_schedule_{ds_name}_{experiment_name}_{mod_name}'

    try:
        table = Table(table_name, metadata, autoload=True)

        found_new_column = False
        for col_name, dtype in current_cleaning_setting.dtypes.items():
            if col_name not in table.columns:
                current_cleaning_setting.to_sql(table_name, con=database_engine, if_exists='replace', index=False)
                found_new_column = True
        if not found_new_column:
            current_cleaning_setting.to_sql(table_name, con=database_engine, if_exists='append', index=False)
    except NoSuchTableError:
        current_cleaning_setting.to_sql(table_name, con=database_engine, if_exists='replace', index=False)


def decode_features_importance_results(ds_name, encoded_features_importance_df, metadata):
    original_shap_values = {}
    for idx, original_feature in enumerate(metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols']):
        encoded_feature_names = [name for i, name in enumerate(encoded_features_importance_df.columns) if
                                 name.startswith(original_feature)]
        original_shap_values[original_feature] = \
        encoded_features_importance_df[encoded_feature_names].sum(axis=1).values[0]
    decoded_features_importance_df = pd.DataFrame(original_shap_values, index=[0])
    decoded_features_importance_df = decoded_features_importance_df.T[0]
    decoded_features_importance_df = decoded_features_importance_df.sort_values(ascending=False)
    return decoded_features_importance_df


def get_features_importance(pre_pollution_df, ds_name, experiment, modifier, metadata, pollution_setting):
    print('Start features importance calculation')
    #return pd.DataFrame(), False
    ap = ArtificialPollution(metadata, str(database_engine.url), pollution_setting['pre_pollution_setting_id'], modifier)
    filtered_history_df, is_empty = ap.get_filtered_history_df(pre_pollution_df, 277712)
    if is_empty:
        warnings.warn(f'Empty pre_pollution_df for {ds_name}, {modifier.__name__} and random_seed 277712', UserWarning)
        return pd.DataFrame(), is_empty

    train_df_polluted, test_df_polluted = ap.get_current_polluted_training_and_test_df(filtered_history_df, ds_name,
                                                                                       pollution_setting)

    exp = experiment(train_df_polluted, test_df_polluted, metadata[ds_name], modifier.get_classname(), pollution_setting['pre_pollution_setting_id'])
    results = exp.run('', 'scenario', explain=True)
    encoded_features_importance_results = results[exp.name]['feature_importances']['fi_over_test']['global']
    encoded_features_importance_df = pd.DataFrame(encoded_features_importance_results, index=[0])

    decoded_features_importance_df = decode_features_importance_results(ds_name, encoded_features_importance_df,
                                                                        metadata)
    print('End features importance calculation')
    return decoded_features_importance_df, is_empty


def set_cleaning_setting(feature, pollution_level, predicted_poly_reg_f1, real_f1, used_budget, f1_gain_predicted, prediction_score):
    cleaning_setting = {}
    cleaning_setting['feature'] = feature
    cleaning_setting['pollution_level'] = pollution_level
    cleaning_setting['predicted_poly_reg_f1'] = predicted_poly_reg_f1
    cleaning_setting['real_f1'] = real_f1
    cleaning_setting['used_budget'] = used_budget
    cleaning_setting['f1_gain_predicted'] = f1_gain_predicted
    cleaning_setting['prediction_score'] = prediction_score
    return cleaning_setting


def main(ml_algorithm, error_type, ds_name: str, original_budget, metadata, database_engine, pre_pollution_setting_ids) -> None:

    table_name = f'cleaning_schedule_{ds_name}_{ml_algorithm.__name__}_{error_type.__name__}'
    delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids)

    pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)
    if pre_pollution_df.empty:
        warnings.warn(f'No pre pollution df found for {ds_name} and {error_type.__name__}. Stopping execution.'
                      f'Please run pre pollution first.', UserWarning)
        return None

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=pre_pollution_setting_ids)
    print(pre_pollution_settings)

    for pollution_setting in pre_pollution_settings:
        # measure execution time for each pollution setting
        start_time = time.time()
        features_importance, _ = get_features_importance(pre_pollution_df, ds_name, ml_algorithm, error_type, metadata, pollution_setting)
        BUDGET = original_budget
        cleaning_schedule = []
        pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
        iteration = 1
        BUDGET = 50
        prediction_accuracy_history = {col: [] for col in metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols']}
        cleaner = DataCleaning(CleaningSetting5())

        while BUDGET > 0:
            print(f'Current config: {ml_algorithm.__name__}, {error_type.__name__}, {ds_name}, iteration {iteration}')

            ap = ArtificialPollution(metadata, str(database_engine.url), pre_pollution_setting_id, error_type)
            polluted_dfs, original_f1_score = ap.artificial_pollution(pre_pollution_df, ds_name, ml_algorithm, pollution_setting, iteration, skip_pollution=False)

            if len(polluted_dfs) == 0:
                print('Nothing to clean anymore.')
                break
            cleaning_setting = cleaner.perform_cleaning(polluted_dfs, pollution_setting, BUDGET,
                                                        prediction_accuracy_history=prediction_accuracy_history,
                                                        features_importance=features_importance)
            if cleaning_setting['feature'] is None:
                print('Nothing to clean anymore.')
                break
            write_cleaning_setting_to_db(cleaning_setting, iteration, ds_name, ml_algorithm.__name__, error_type.__name__, original_f1_score, original_budget, pre_pollution_setting_id, database_engine)

            cleaning_schedule.append(cleaning_setting)
            print(f'iteration {iteration}; cleaning_setting', cleaning_setting)
            pollution_setting = update_feature_wise_pollution_level(pollution_setting, cleaning_setting)
            print('feature_wise_pollution_level', pollution_setting)

            BUDGET = BUDGET - cleaning_setting['used_budget']
            print('Budget: ', BUDGET)

            iteration += 1
            print('cleaning_schedule', cleaning_schedule)

            if len(cleaner.get_cleaning_buffer()) > 0 and BUDGET == 0:
                # buffer entries string representation, seperated by comma
                f1_after_cleaning = ap.clean_from_cleaning_buffer(cleaner.get_cleaning_buffer(), pre_pollution_df, ds_name, ml_algorithm, pollution_setting)
                buffer_entries_str = ', '.join([str(entry) for entry in cleaner.get_cleaning_buffer()])
                cleaning_setting = set_cleaning_setting(f'<BUFFER> ({buffer_entries_str})', 0.0, f1_after_cleaning, f1_after_cleaning, 0.0, 0.0, f1_after_cleaning)
                write_cleaning_setting_to_db(cleaning_setting, iteration, ds_name, ml_algorithm.__name__, error_type.__name__, cleaning_setting, original_budget, pre_pollution_setting_id, database_engine)

        print('cleaning_schedule', cleaning_schedule)
        current_time = time.time()
        print('Needed time for all pre-pollution settings', (current_time - start_time) / (60 * 60), 'hours')
        # write time measurement to db
        table_name = 'dynamic_greedy_time_measurements'
        #write_time_measurement_to_db(table_name, ds_name, ml_algorithm.__name__, error_type.__name__, current_time - start_time, pre_pollution_setting_id, database_engine)
    print('Finished dynamic greedy for', ds_name, ml_algorithm.__name__, error_type.__name__)


if __name__ == "__main__":

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
    parser.add_argument('--ml_algorithm', default='SupportVectorMachineExperiment', type=str, help='Set the ML algorithm to use for the experiment.')
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
    # space separated string of pre pollution setting ids to list of ints
    pre_pollution_setting_ids = [int(i) for i in args.pre_pollution_settings.split(' ')]

    database_engine = create_engine(database_url, echo=True, connect_args={'timeout': 1000})

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    main(chosen_ml_algorithm, chosen_error_type, ds_name, budget, metadata, database_engine, pre_pollution_setting_ids)
