import hashlib
import time

import numpy as np
from sqlalchemy import create_engine
from classification.utils.DatasetModifier import *
from classification.utils.artifical_pollution import *
from classification.experiments import *
from json import load as load_json
from classification.utils.util import load_pre_pollution_df, get_pre_pollution_settings, delete_entries_from_table
from classification.pre_pollution import generate_multi_error_data
import random
from util import start_logging
import argparse
import os
from config.definitions import ROOT_DIR
from classification.utils.ExperimentRunner import ExperimentRunner
from classification.utils.data_cleaning import SimulationCleaning, Cleaner
from classification.utils.cleaning_recommendation import CometRecommendationStrategy, Recommender, CleaningResult
pd.options.mode.chained_assignment = None


def write_cleaning_setting_to_db(cleaning_setting, iteration, ds_name, experiment_name, mod_name, original_f1_score, original_budget, pre_pollution_setting_id, database_engine, run):
    current_cleaning_setting = cleaning_setting.copy()
    current_cleaning_setting['iteration'] = iteration
    current_cleaning_setting['dataset'] = ds_name
    current_cleaning_setting['experiment'] = experiment_name
    current_cleaning_setting['polluter'] = mod_name
    current_cleaning_setting['original_f1_score'] = original_f1_score
    current_cleaning_setting['original_budget'] = original_budget
    current_cleaning_setting['feature'] = current_cleaning_setting['feature']
    current_cleaning_setting['pre_pollution_setting_id'] = pre_pollution_setting_id
    current_cleaning_setting['run'] = run

    current_cleaning_setting = pd.DataFrame(current_cleaning_setting, index=[0])
    table_name = f'cleaning_schedule_completely_random_{ds_name}_{experiment_name}'
    #current_cleaning_setting.to_sql(table_name, con=database_engine, if_exists='append', index=False)
    with open(f'{ROOT_DIR}/slurm/completely_random/RESULTS/{table_name}_{pre_pollution_setting_id}.csv', 'a') as f:
        # if file is empty then write header
        if os.stat(f'{ROOT_DIR}/slurm/completely_random/RESULTS/{table_name}_{pre_pollution_setting_id}.csv').st_size == 0:
            current_cleaning_setting.to_csv(f, header=True, index=False)
        else:
            current_cleaning_setting.to_csv(f, header=False, index=False)


def main(ml_algorithm, error_type, ds_name, original_budget, metadata, database_engine, pre_pollution_setting_ids):
    table_name = f'cleaning_schedule_completely_random_{ds_name}_{ml_algorithm.__name__}'
    #delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids)

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=pre_pollution_setting_ids)
    print(pre_pollution_settings)
    error_types = [MissingValuesModifier, CategoricalShiftModifier, ScalingModifier, GaussianNoiseModifier]

    for pre_pollution_setting in pre_pollution_settings:
        pre_pollution_setting_id = pre_pollution_setting['pre_pollution_setting_id']
        if not os.path.exists(f'{ROOT_DIR}/slurm/completely_random/RESULTS/'):
            os.makedirs(f'{ROOT_DIR}/slurm/completely_random/RESULTS/')
        else:
            if os.path.exists(f'{ROOT_DIR}/slurm/completely_random/RESULTS/{table_name}_{pre_pollution_setting_id}.csv'):
                os.remove(f'{ROOT_DIR}/slurm/completely_random/RESULTS/{table_name}_{pre_pollution_setting_id}.csv')

        for run in range(0, 5):
            # reload pre_pollution_setting
            pollution_setting = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=[pre_pollution_setting["pre_pollution_setting_id"]])[0]
            print(f'Run {run} for pollution setting {pollution_setting["pre_pollution_setting_id"]}')
            # pollution_setting = {'pre_pollution_setting_id': pre_pollution_setting_id,
            #                      'train': {'Number of children': 0.05, 'Wife\'s age': 0.01},
            #                      'test': {'Number of children': 0.05, 'Wife\'s age': 0.01}}

            start_time = time.time()
            cleaning_schedule = []
            pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
            results_df = pd.DataFrame(columns=['pollution_level', 'real_f1', 'predicted_f1'])
            iteration = 1
            BUDGET = 50
            error_map_test_df, error_map_train_df, test_df, test_df_polluted, train_df, train_df_polluted = generate_multi_error_data(
                database_engine, ds_name, error_types, metadata, pollution_setting, pre_pollution_setting_id)

            simulation_cleaner = SimulationCleaning()
            cleaner = Cleaner(simulation_cleaner)
            comet_recommendation = CometRecommendationStrategy(write_to_db=True, cleaner=cleaner)
            recommender = Recommender(comet_recommendation)
            er = ExperimentRunner({}, ml_algorithm, metadata=metadata, ds_name=ds_name,
                                  error_type=error_type, pre_pollution_setting_id=pre_pollution_setting_id)

            result = er.parallel_model_performance_calculation(87263, -1.0, ml_algorithm, train_df_polluted, test_df_polluted)
            results_df = pd.concat([results_df, result], ignore_index=True)

            cleaning_setting = {}
            cleaning_setting['feature'] = ''
            cleaning_setting['real_f1'] = results_df['real_f1'].values[0]
            cleaning_setting['error_type'] = ''
            cleaning_setting['used_budget'] = 0

            write_cleaning_setting_to_db(cleaning_setting, 0, ds_name, ml_algorithm.__name__,
                                         '', results_df['real_f1'].values[0], original_budget,
                                         pre_pollution_setting_id, database_engine, run)
            cleaning_schedule.append(cleaning_setting)

            random.seed(run + time.time())

            while BUDGET > 0:
                print(f'Current config: {ml_algorithm.__name__}, {error_type.__name__}, {ds_name}, iteration {iteration}')
                feature_candidates_for_cleaning = metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols']

                counter = 0
                feature = random.choice(feature_candidates_for_cleaning)
                error_type = random.choice(error_types)
                tried_features_error_combi = {}
                while (recommender.check_if_feature_detected_as_clean_before(feature, error_type.__name__) or
                       recommender.cost_function.calculate_used_budget(feature, error_type.__name__, fitting_check=True) > BUDGET):
                    hash_input = str(counter)
                    random_seed = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                    random.seed(random_seed)
                    feature = random.choice(feature_candidates_for_cleaning)
                    error_type = random.choice(error_types)
                    counter += 1
                    tried_features_error_combi[f'{feature}_{error_type.__name__}'] = True
                    if len(tried_features_error_combi.keys()) == len(feature_candidates_for_cleaning)*len(error_types):
                        print('All features are clean, stopping...')
                        break

                cleaning_candidates_train = error_map_train_df[error_map_train_df[feature] == error_type.__name__].index.tolist()
                cleaning_candidates_test = error_map_test_df[error_map_test_df[feature] == error_type.__name__].index.tolist()

                if len(cleaning_candidates_train) == 0 and len(cleaning_candidates_test) == 0:
                    recommender.set_feature_as_cleaned(feature, error_type.__name__)
                    continue

                cleaning_configs = {'data': [],
                                    'cleaning_candidates_train': cleaning_candidates_train,
                                    'cleaning_candidates_test': cleaning_candidates_test}
                cleaned_dfs = cleaner.clean(feature, train_df_polluted=train_df_polluted, test_df_polluted=test_df_polluted,
                                   train_df=train_df, test_df=test_df, cleaning_config=cleaning_configs,
                                   cleaning_step_size=0.01, error_map_train=error_map_train_df, error_map_test=error_map_test_df, error_type=error_type)

                if cleaned_dfs['train']['indexes'] is None and cleaned_dfs['test']['indexes'] is None:
                    recommender.set_feature_as_cleaned(feature, error_type.__name__)
                    print('Feature', feature, 'is clean, skipping...')
                    continue
                train_df_polluted = cleaned_dfs['train']['data']
                test_df_polluted = cleaned_dfs['test']['data']
                result = er.parallel_model_performance_calculation(87263, -1.0, ml_algorithm, train_df_polluted, test_df_polluted)
                f1_score_after_cleaning = float(result['real_f1'].values[0])

                cleaning_setting = {}
                cleaning_setting['feature'] = feature
                cleaning_setting['real_f1'] = f1_score_after_cleaning
                cleaning_setting['error_type'] = error_type.__name__
                cleaning_setting['used_budget'] = recommender.cost_function.calculate_used_budget(feature, error_type.__name__)

                write_cleaning_setting_to_db(cleaning_setting, iteration, ds_name, ml_algorithm.__name__,
                                             error_type.__name__, results_df['real_f1'].values[0], original_budget,
                                             pre_pollution_setting_id, database_engine, run)

                cleaning_schedule.append(cleaning_setting)

                if cleaned_dfs['train']['indexes'] is not None:
                    error_map_train_df.loc[cleaned_dfs['train']['indexes'], feature] = 'correct'
                if cleaned_dfs['test']['indexes'] is not None:
                    error_map_test_df.loc[cleaned_dfs['test']['indexes'], feature] = 'correct'
                if error_map_train_df.isin(['correct']).all().all() and error_map_test_df.isin(['correct']).all().all():
                    print('All features are clean, stopping...')
                    break

                print(f'iteration {iteration}; cleaning_setting', cleaning_setting)
                BUDGET = BUDGET - cleaning_setting['used_budget']
                iteration += 1
                print('cleaning_schedule', cleaning_schedule)
                print(f'Needed time for current pre-pollution setting {pre_pollution_setting_id}', (time.time() - start_time), 'seconds')

            print('cleaning_schedule', cleaning_schedule)
            print('Needed time for all pre-pollution settings', (time.time() - start_time), 'seconds')


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
    # space separated string of pre pollution setting ids to list of ints
    pre_pollution_setting_ids = [int(i) for i in args.pre_pollution_settings.split(' ')]

    database_engine = create_engine(database_url, echo=True, connect_args={'timeout': 1000})

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    main(chosen_ml_algorithm, chosen_error_type, ds_name, budget, metadata, database_engine, pre_pollution_setting_ids)
