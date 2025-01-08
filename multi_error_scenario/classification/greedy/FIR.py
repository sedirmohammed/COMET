import hashlib
import random
import time
from json import load as load_json
from classification.utils.DatasetModifier import *
from classification.experiments import *
from classification.utils.util import get_pre_pollution_settings, load_pre_pollution_df, delete_entries_from_table
from util import start_logging
from sqlalchemy import create_engine
import argparse
from config.definitions import ROOT_DIR
import os
from classification.utils.artifical_pollution import *
from classification.pre_pollution import generate_multi_error_data
from classification.utils.ExperimentRunner import ExperimentRunner
from classification.utils.data_cleaning import SimulationCleaning, Cleaner
from classification.utils.cleaning_recommendation import CometRecommendationStrategy, Recommender, CleaningResult


def get_f1_score_results(pre_pollution_df, ds_name, experiment, modifier, metadata, pollution_setting):

    ap = ArtificialPollution(metadata, str(database_engine.url), pollution_setting['pre_pollution_setting_id'], modifier)
    filtered_history_df, is_empty = ap.get_filtered_history_df(pre_pollution_df, 277712)
    if is_empty:
        warnings.warn(f'Empty pre_pollution_df for {ds_name}, {modifier.__name__} and random_seed 277712', UserWarning)
        return pd.DataFrame(), is_empty

    train_df_polluted, test_df_polluted = ap.get_current_polluted_training_and_test_df(filtered_history_df, ds_name,
                                                                                       pollution_setting)

    f1_score_results = Parallel(n_jobs=1)(
        delayed(parallel_model_performance_calculation)(random_seed, experiment, train_df_polluted, test_df_polluted, ds_name, modifier, pollution_setting['pre_pollution_setting_id'])
        for random_seed in [87263])
    return np.mean(f1_score_results)


def parallel_model_performance_calculation(random_seed, experiment, train_df_polluted, test_df_polluted, ds_name, modifier, pre_pollution_setting_id):
    np.random.seed(random_seed)
    exp = experiment(train_df_polluted, test_df_polluted, metadata[ds_name], modifier.get_classname(), pre_pollution_setting_id)
    results = exp.run('', 'scenario', explain=False)
    return results[exp.name]['scoring']['macro avg']['f1-score']


def get_features_importance(ds_name, experiment, error_types, metadata, pollution_setting):
    # data = {
    #     "Wife's now working?": 0.017100,
    #     "Standard-of-living index": 0.005326,
    #     "Wife's religion": -0.000214,
    #     "Media exposure": -0.000542,
    #     "Husband's education": -0.000570,
    #     "Wife's age": -0.000869,
    #     "Husband's occupation": -0.003896,
    #     "Number of children": -0.009439,
    #     "Wife's education": -0.050286
    # }
    # series = pd.Series(data, name='0', dtype='float64')
    # return series

    error_map_test_df, error_map_train_df, test_df, test_df_polluted, train_df, train_df_polluted = generate_multi_error_data(
        database_engine, ds_name, error_types, metadata, pollution_setting, pollution_setting['pre_pollution_setting_id'])

    exp = experiment(train_df_polluted, test_df_polluted, metadata[ds_name], 'multi_error', pollution_setting['pre_pollution_setting_id'])
    results = exp.run('', 'scenario', explain=True)
    encoded_features_importance_results = results[exp.name]['feature_importances']['fi_over_test']['global']
    encoded_features_importance_df = pd.DataFrame(encoded_features_importance_results, index=[0])

    decoded_features_importance_df = decode_features_importance_results(ds_name, encoded_features_importance_df,
                                                                        metadata)
    return decoded_features_importance_df


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


def main(ml_algorithm, error_type, ds_name, original_budget, metadata, database_engine, pre_pollution_setting_ids) -> None:

    table_name = f'cleaning_schedule_static_features_importance_{ds_name}_{ml_algorithm.__name__}'
    #delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids)

    pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)
    error_types = [MissingValuesModifier, CategoricalShiftModifier, ScalingModifier, GaussianNoiseModifier]

    if pre_pollution_df.empty:
        warnings.warn(f'No pre pollution df found for {ds_name} and {error_type}. Stopping execution.'
                      f'Please run pre pollution first.', UserWarning)
        return

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=pre_pollution_setting_ids)
    for pollution_setting in pre_pollution_settings:
        start_time = time.time()
        pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
        if not os.path.exists(f'{ROOT_DIR}/slurm/static_fi_greedy/RESULTS/'):
            os.makedirs(f'{ROOT_DIR}/slurm/static_fi_greedy/RESULTS/')
        else:
            if os.path.exists(f'{ROOT_DIR}/slurm/static_fi_greedy/RESULTS/{table_name}_{pre_pollution_setting_id}.csv'):
                os.remove(f'{ROOT_DIR}/slurm/static_fi_greedy/RESULTS/{table_name}_{pre_pollution_setting_id}.csv')
                time.sleep(30)
                if os.path.exists(f'{ROOT_DIR}/slurm/static_fi_greedy/RESULTS/{table_name}_{pre_pollution_setting_id}.csv'):
                    warnings.warn(f'Could not delete {ROOT_DIR}/slurm/static_fi_greedy/RESULTS/{table_name}_{pre_pollution_setting_id}.csv', UserWarning)
                    return
        #pollution_setting = {'pre_pollution_setting_id': pre_pollution_setting_id, 'train': {'Number of children': 0.10, 'Wife\'s age': 0.01}, 'test': {'Number of children': 0.10, 'Wife\'s age': 0.01}}

        print(pollution_setting)
        cleaning_schedule = []
        iteration = 0
        results_df = pd.DataFrame(columns=['pollution_level', 'real_f1', 'predicted_f1'])
        simulation_cleaner = SimulationCleaning()
        cleaner = Cleaner(simulation_cleaner)
        comet_recommendation = CometRecommendationStrategy(write_to_db=True, cleaner=cleaner)
        recommender = Recommender(comet_recommendation)
        er = ExperimentRunner({}, ml_algorithm, metadata=metadata, ds_name=ds_name,
                              error_type=error_type, pre_pollution_setting_id=pre_pollution_setting_id)
        error_map_test_df, error_map_train_df, test_df, test_df_polluted, train_df, train_df_polluted = generate_multi_error_data(
            database_engine, ds_name, error_types, metadata, pollution_setting, pre_pollution_setting_id)
        result = er.parallel_model_performance_calculation(87263, -1.0, ml_algorithm, train_df_polluted, test_df_polluted)
        results_df = pd.concat([results_df, result], ignore_index=True)

        cleaning_setting = {}
        cleaning_setting['feature'] = ''
        cleaning_setting['real_f1'] = result['real_f1'].values[0]
        cleaning_setting['error_type'] = ''
        cleaning_setting['used_budget'] = 0
        cleaning_schedule.append(cleaning_setting)
        write_cleaning_schedule_to_db(cleaning_setting, ds_name, ml_algorithm.__name__, pre_pollution_setting_id)

        exp = ml_algorithm(train_df, test_df, metadata[ds_name], 'multi_error', pre_pollution_setting_id)
        results = exp.run('', 'scenario', explain=False)
        print('clean F1 score', results[exp.name]['scoring']['macro avg']['f1-score'])

        features_importance = get_features_importance(ds_name, ml_algorithm, error_types, metadata, pollution_setting)
        feature_candidates_for_cleaning = metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols']
        tried_feature_error_combi = {}
        for feature in feature_candidates_for_cleaning:
            tried_feature_error_combi[feature] = {}
        importance_pos = 0
        BUDGET = 50
        feature = None
        error_type = None
        while BUDGET > 0:
            if feature is None or error_type is None:
                hash_input = str(pre_pollution_setting_id) + ds_name
                random_seed = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                random.seed(random_seed)
                error_type = random.choice(error_types)
                feature = features_importance.index[importance_pos]
                tried_feature_error_combi[feature][error_type.__name__] = True
            print(f'Current config: {pollution_setting}, iteration: {iteration}, {ds_name}, {ml_algorithm.__name__}, {error_type.__name__}')
            counter = 0
            while (recommender.check_if_feature_detected_as_clean_before(feature, error_type.__name__) or
                   recommender.cost_function.calculate_used_budget(feature, error_type.__name__, fitting_check=True) > BUDGET):
                if len(tried_feature_error_combi[feature].keys()) == len(error_types):
                    importance_pos += 1
                    if importance_pos == len(features_importance):
                        print('All features are clean, stopping...')
                        quit()
                    feature = features_importance.index[importance_pos]
                hash_input = str(counter)
                random_seed = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                random.seed(random_seed)
                error_type = random.choice(error_types)
                tried_feature_error_combi[feature][error_type.__name__] = True
                counter += 1

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
                                        cleaning_step_size=0.01, error_map_train=error_map_train_df,
                                        error_map_test=error_map_test_df, error_type=error_type)

            if cleaned_dfs['train']['indexes'] is None and cleaned_dfs['test']['indexes'] is None:
                recommender.set_feature_as_cleaned(feature, error_type.__name__)
                print('Feature', feature, 'is clean, skipping...')
                continue
            train_df_polluted = cleaned_dfs['train']['data']
            test_df_polluted = cleaned_dfs['test']['data']
            result = er.parallel_model_performance_calculation(87263, -1.0, ml_algorithm, train_df_polluted, test_df_polluted)
            results_df = pd.concat([results_df, result], ignore_index=True)
            f1_score_after_cleaning = float(result['real_f1'].values[0])

            cleaning_setting = {}
            cleaning_setting['feature'] = feature
            cleaning_setting['real_f1'] = f1_score_after_cleaning
            cleaning_setting['error_type'] = error_type.__name__
            cleaning_setting['used_budget'] = recommender.cost_function.calculate_used_budget(feature, error_type.__name__)
            cleaning_schedule.append(cleaning_setting)

            write_cleaning_schedule_to_db(cleaning_setting, ds_name, ml_algorithm.__name__, pre_pollution_setting_id)

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
            #importance_pos += 1
            print('cleaning_schedule', cleaning_schedule)
            print(f'Needed time for current pre-pollution setting {pre_pollution_setting_id}', (time.time() - start_time), 'seconds')


def check_feature_type_match(ds_name, features_importance, mod, k):
    if features_importance.index[k] in metadata[ds_name]['categorical_cols']:
        feature_type = 'categorical_col'
    else:
        feature_type = 'numerical_col'
    if feature_type != mod.restricted_to and mod.restricted_to != '':
        return False
    else:
        return True


def write_cleaning_schedule_to_db(cleaning_setting, ds_name, experiment_name, pre_pollution_setting_id):
    cleaning_setting['pre_pollution_setting_id'] = pre_pollution_setting_id
    current_cleaning_setting_df = pd.DataFrame(cleaning_setting, index=[0])
    table_name = f'cleaning_schedule_static_features_importance_{ds_name}_{experiment_name}'
    #current_cleaning_setting_df.to_sql(table_name, database_engine, if_exists='append', index=False)
    with open(f'{ROOT_DIR}/slurm/static_fi_greedy/RESULTS/{table_name}_{pre_pollution_setting_id}.csv', 'a') as f:
        # if file is empty then write header
        if os.stat(f'{ROOT_DIR}/slurm/static_fi_greedy/RESULTS/{table_name}_{pre_pollution_setting_id}.csv').st_size == 0:
            current_cleaning_setting_df.to_csv(f, header=True, index=False)
        else:
            current_cleaning_setting_df.to_csv(f, header=False, index=False)


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
