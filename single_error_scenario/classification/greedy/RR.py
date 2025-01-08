import time
from sqlalchemy import create_engine
from classification.utils.DatasetModifier import *
from classification.utils.artifical_pollution import *
from classification.experiments import *
from json import load as load_json
from classification.utils.util import load_pre_pollution_df, get_pre_pollution_settings, delete_entries_from_table
import random
from util import start_logging
import argparse
import os
from config.definitions import ROOT_DIR
pd.options.mode.chained_assignment = None


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
    table_name = f'cleaning_schedule_completely_random_{ds_name}_{experiment_name}_{mod_name}'
    with open(f'{ROOT_DIR}/slurm/completely_random/RESULTS/{table_name}_{pre_pollution_setting_id}.csv', 'a') as f:
        if os.stat(f'{ROOT_DIR}/slurm/completely_random/RESULTS/{table_name}_{pre_pollution_setting_id}.csv').st_size == 0:
            current_cleaning_setting.to_csv(f, header=True, index=False)
        else:
            current_cleaning_setting.to_csv(f, header=False, index=False)


def main(ml_algorithm, error_type, ds_name, original_budget, metadata, database_engine, pre_pollution_setting_ids):
    table_name = f'cleaning_schedule_completely_random_{ds_name}_{ml_algorithm.__name__}_{error_type.__name__}'
    #delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids)

    pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=pre_pollution_setting_ids)
    print(pre_pollution_settings)

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

            start_time = time.time()
            cleaning_schedule = []
            pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
            iteration = 1
            BUDGET = 50
            while BUDGET > 0:
                print(f'Current config: {ml_algorithm.__name__}, {error_type.__name__}, {ds_name}, iteration {iteration}')

                ap = ArtificialPollution(metadata, str(database_engine.url), pollution_setting['pre_pollution_setting_id'], error_type)

                feature_candidates_for_cleaning = metadata[ds_name]['categorical_cols'] + metadata[ds_name]['numerical_cols']
                feature_candidates_for_cleaning = ap.filter_features_on_type(feature_candidates_for_cleaning, metadata, ds_name)
                new_feature_candidates_for_cleaning = []
                for feature in feature_candidates_for_cleaning:
                    current_pollution_level_train = pollution_setting['train'][feature]
                    current_pollution_level_test = pollution_setting['test'][feature]
                    current_pollution_level = {'train': current_pollution_level_train,
                                               'test': current_pollution_level_test}

                    if current_pollution_level['train'] == 0 and current_pollution_level['test'] == 0:
                        continue
                    else:
                        new_feature_candidates_for_cleaning.append(feature)
                if len(new_feature_candidates_for_cleaning) == 0:
                    print('Nothing to clean anymore.')
                    break

                feature_entry = random.choice(new_feature_candidates_for_cleaning)
                filtered_history_df, is_empty = ap.get_filtered_history_df(pre_pollution_df, 277712)
                if pollution_setting['train'][feature_entry] > 0:
                    pollution_setting['train'][feature_entry] = round(pollution_setting['train'][feature_entry] - 0.01, 2)
                if pollution_setting['test'][feature_entry] > 0:
                    pollution_setting['test'][feature_entry] = round(pollution_setting['test'][feature_entry] - 0.01, 2)
                train_df_polluted, test_df_polluted = ap.get_current_polluted_training_and_test_df(filtered_history_df, ds_name, pollution_setting)
                print(f'New pollution level for {feature_entry} is {pollution_setting["train"][feature_entry]}')
                result = Parallel(n_jobs=1)(
                    delayed(ap.parallel_model_performance_calculation)(random_seed, pollution_setting['train'][feature_entry], ml_algorithm,
                                                                         train_df_polluted, test_df_polluted, ds_name)
                    for random_seed in [87263, 53219, 78604, 2023, 38472, 11, 9834, 4567, 909090, 56789])
                    #for random_seed in [87263])
                results_df = pd.concat(result)

                cleaning_setting = {}
                cleaning_setting['feature'] = feature_entry
                cleaning_setting['pollution_level'] = results_df['pollution_level'].values[0]
                cleaning_setting['predicted_poly_reg_f1'] = -1
                cleaning_setting['real_f1'] = results_df['real_f1'].values[0]
                cleaning_setting['used_budget'] = 1
                cleaning_setting['f1_gain_predicted'] = -1

                write_cleaning_setting_to_db(cleaning_setting, iteration, ds_name, ml_algorithm.__name__, error_type.__name__, results_df['real_f1'].values[0], original_budget, pre_pollution_setting_id, database_engine)

                cleaning_schedule.append(cleaning_setting)
                print(f'iteration {iteration}; cleaning_setting', cleaning_setting)

                #pollution_setting = update_feature_wise_pollution_level(pollution_setting, cleaning_setting)
                print('feature_wise_pollution_level', pollution_setting)

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
    pre_pollution_setting_ids = [int(i) for i in args.pre_pollution_settings.split(' ')]

    database_engine = create_engine(database_url, echo=True, connect_args={'timeout': 1000})

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    main(chosen_ml_algorithm, chosen_error_type, ds_name, budget, metadata, database_engine, pre_pollution_setting_ids)
