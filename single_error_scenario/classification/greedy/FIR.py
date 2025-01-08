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


def get_features_importance(pre_pollution_df, ds_name, experiment, modifier, metadata, pollution_setting):

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
    return decoded_features_importance_df, is_empty


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

    table_name = f'cleaning_schedule_static_features_importance_{ds_name}_{ml_algorithm.__name__}_{error_type.__name__}'
    #delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids)

    pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)
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

        print(pollution_setting)
        cleaning_schedule = []
        iteration = 0
        previous_cleaning_setting = {}
        features_importance, filtered_pre_pollution_df_is_empty = get_features_importance(pre_pollution_df, ds_name, ml_algorithm, error_type, metadata, pollution_setting)
        if filtered_pre_pollution_df_is_empty:
            continue
        BUDGET = 50
        while BUDGET > 0:
            print(f'Current config: {pollution_setting}, iteration: {iteration}, {ds_name}, {ml_algorithm.__name__}, {error_type.__name__}')

            k = 0
            f1_score_results_avg = get_f1_score_results(pre_pollution_df, ds_name, ml_algorithm, error_type, metadata, pollution_setting)
            print(f'f1_score_results_avg: {f1_score_results_avg}, for iteration: {iteration}')

            try:
                while (pollution_setting['train'][features_importance.index[k]] == 0.0 and pollution_setting['test'][features_importance.index[k]] == 0.0) or check_feature_type_match(ds_name, features_importance, error_type, k) == False:
                    k += 1
                    continue
            except IndexError:
                print('No more features to clean')
                break

            selected_feature_for_cleaning = features_importance.index[k]

            number_of_cleaning = 0
            used_budget = 0
            while used_budget <= metadata['pre_pollution_level']*100 and BUDGET > 0 and (pollution_setting['train'][selected_feature_for_cleaning] != 0.0 or pollution_setting['test'][selected_feature_for_cleaning] != 0.0) and number_of_cleaning < 1:
                if pollution_setting['train'][selected_feature_for_cleaning] != 0.0:
                    pollution_setting['train'][selected_feature_for_cleaning] -= 0.01
                    pollution_setting['train'][selected_feature_for_cleaning] = round(pollution_setting['train'][selected_feature_for_cleaning], 2)
                if pollution_setting['test'][selected_feature_for_cleaning] != 0.0:
                    pollution_setting['test'][selected_feature_for_cleaning] -= 0.01
                    pollution_setting['test'][selected_feature_for_cleaning] = round(pollution_setting['test'][selected_feature_for_cleaning], 2)
                used_budget += 1
                BUDGET -= 1
                number_of_cleaning += 1

            cleaning_setting = {'feature': selected_feature_for_cleaning,
                                'pollution_level': f'train: {pollution_setting["train"][selected_feature_for_cleaning]}, test: {pollution_setting["test"][selected_feature_for_cleaning]}',
                                'used_budget': used_budget,
                                'iteration': iteration,
                                'f1_score': f1_score_results_avg,
                                'polluter': error_type.__name__,
                                'dataset': ds_name,
                                'experiment': ml_algorithm.__name__,
                                'original_budget': original_budget}

            if iteration > 0:
                previous_feature = previous_cleaning_setting['feature']
                previous_pollution_level = previous_cleaning_setting['pollution_level']
                previous_used_budget = previous_cleaning_setting['used_budget']

            previous_cleaning_setting = cleaning_setting.copy()

            if iteration == 0:
                cleaning_setting['feature'] = ''
                cleaning_setting['pollution_level'] = None
                cleaning_setting['used_budget'] = None
            else:
                cleaning_setting['feature'] = previous_feature
                cleaning_setting['pollution_level'] = previous_pollution_level
                cleaning_setting['used_budget'] = previous_used_budget

            write_cleaning_schedule_to_db(cleaning_setting, ds_name, ml_algorithm.__name__, error_type.__name__, pre_pollution_setting_id, database_engine)
            cleaning_schedule.append(cleaning_setting)
            iteration += 1
            print(cleaning_setting)

        f1_score_results_avg = get_f1_score_results(pre_pollution_df, ds_name, ml_algorithm, error_type, metadata, pollution_setting)
        if len(previous_cleaning_setting) == 0:
            warnings.warn(f'No feature was ever considered for cleaning for this considered pre pollution setting. Skipping this setting.', UserWarning)
            continue
        previous_feature = previous_cleaning_setting['feature']
        previous_pollution_level = previous_cleaning_setting['pollution_level']
        previous_used_budget = previous_cleaning_setting['used_budget']

        cleaning_setting['feature'] = previous_feature
        cleaning_setting['pollution_level'] = previous_pollution_level
        cleaning_setting['used_budget'] = previous_used_budget
        cleaning_setting['iteration'] = iteration
        cleaning_setting['f1_score'] = f1_score_results_avg

        write_cleaning_schedule_to_db(cleaning_setting, ds_name, ml_algorithm.__name__, error_type.__name__, pre_pollution_setting_id, database_engine)
        cleaning_schedule.append(cleaning_setting)

        current_time = time.time()
        print('Needed time for current pre-pollution setting', (current_time - start_time) * 60, 'seconds')
        table_name = 'static_feature_importance_greedy_time_measurements'
        #write_time_measurement_to_db(table_name, ds_name, ml_algorithm.__name__, error_type.__name__,
        #                             current_time - start_time, pre_pollution_setting_id, database_engine)


def check_feature_type_match(ds_name, features_importance, mod, k):
    if features_importance.index[k] in metadata[ds_name]['categorical_cols']:
        feature_type = 'categorical_col'
    else:
        feature_type = 'numerical_col'
    if feature_type != mod.restricted_to and mod.restricted_to != '':
        return False
    else:
        return True


def write_cleaning_schedule_to_db(cleaning_setting, ds_name, experiment_name, mod_name, pre_pollution_setting_id, database_engine):
    cleaning_setting['pre_pollution_setting_id'] = pre_pollution_setting_id
    current_cleaning_setting_df = pd.DataFrame(cleaning_setting, index=[0])
    table_name = f'cleaning_schedule_static_features_importance_{ds_name}_{experiment_name}_{mod_name}'
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
