import time

from classification.utils.DatasetModifier import *
from classification.utils.artifical_pollution_new import *
from classification.experiments import *
from json import load as load_json

from classification.utils.cleaning_config import CleaningConfig
from classification.utils.cleaning_recommendation import CometRecommendationStrategy, Recommender, CleaningResult
from classification.utils.data_cleaning import SimulationCleaning, Cleaner
from classification.utils.util import load_pre_pollution_df, get_pre_pollution_settings, delete_entries_from_table
import warnings
import argparse
import os
from sqlalchemy import create_engine, MetaData, Table
from classification.utils.ExperimentRunner import ExperimentRunner
from classification.utils.RegressionRunner import RegressionRunner

from config.definitions import ROOT_DIR
from typing import Dict, List
pd.options.mode.chained_assignment = None


def main(ml_algorithm, error_type, ds_name: str, original_budget, metadata, database_engine, pre_pollution_setting_ids) -> None:

    pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)
    if pre_pollution_df.empty:
        warnings.warn(f'No pre pollution df found for {ds_name} and {error_type.__name__}. Stopping execution.'
                      f'Please run pre pollution first.', UserWarning)
        return None

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=pre_pollution_setting_ids)
    print(pre_pollution_settings)

    time_tracker = {}
    for pollution_setting in pre_pollution_settings:
        print(f'Start dynamic greedy for {ds_name}, {ml_algorithm.__name__}, {error_type.__name__}, pre-pollution setting {pollution_setting["pre_pollution_setting_id"]}')
        # measure execution time for each pollution setting
        start_time = time.time()
        BUDGET = original_budget
        pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
        iteration = 1

        ap = ArtificialPollution2(metadata, str(database_engine.url), pre_pollution_setting_id, error_type, pre_pollution_df, ds_name, pollution_setting)
        train_df = ap.train_df
        test_df = ap.test_df
        train_df_polluted = ap.train_df_polluted
        test_df_polluted = ap.test_df_polluted

        if ml_algorithm_str.startswith('SGDClassifier'):
            label_encoder = LabelEncoder()
            train_df[metadata[ds_name]['target']] = label_encoder.fit_transform(train_df[metadata[ds_name]['target']])
            test_df[metadata[ds_name]['target']] = label_encoder.transform(test_df[metadata[ds_name]['target']])
            train_df_polluted[metadata[ds_name]['target']] = label_encoder.transform(train_df_polluted[metadata[ds_name]['target']])
            test_df_polluted[metadata[ds_name]['target']] = label_encoder.transform(test_df_polluted[metadata[ds_name]['target']])

        simulation_cleaner = SimulationCleaning()
        cleaner = Cleaner(simulation_cleaner)
        comet_recommendation = CometRecommendationStrategy(write_to_db=False, cleaner=cleaner)
        recommender = Recommender(comet_recommendation)

        exp = ml_algorithm(train_df_polluted, test_df_polluted, metadata[ds_name], error_type.get_classname(), pre_pollution_setting_id)
        results = exp.run('', 'scenario', explain=False)
        print(results[exp.name]['scoring']['macro avg']['f1-score'])

        exp = ml_algorithm(train_df, test_df, metadata[ds_name], error_type.get_classname(), pre_pollution_setting_id)
        results = exp.run('', 'scenario', explain=False)
        print('clean F1 score', results[exp.name]['scoring']['macro avg']['f1-score'])


        print(f'Start with actual cleaning process for {ds_name}, {ml_algorithm.__name__}, {error_type.__name__}, pre-pollution setting {pre_pollution_setting_id}')
        while iteration <= 1:
            print(f'Current config: {ml_algorithm.__name__}, {error_type.__name__}, {ds_name}, iteration {iteration}')

            pollution_results: Dict[str, List[PollutedFeature]] = ap.artificial_pollution(train_df_polluted, test_df_polluted, ds_name)
            er = ExperimentRunner(pollution_results, ml_algorithm, metadata=metadata, ds_name=ds_name, error_type=error_type, pre_pollution_setting_id=pre_pollution_setting_id)
            experiment_results: Dict[str, DataFrame] = er.run()

            rr = RegressionRunner(experiment_results)
            regression_results = rr.run()

            cleaning_config_obj = CleaningConfig(regression_results, pollution_results)
            cleaning_configs = cleaning_config_obj.get_configs()

            cleaning_result: CleaningResult = recommender.recommend_and_clean(cleaning_configs=cleaning_configs,
                                                                              train_df=train_df, test_df=test_df,
                                                                              train_df_polluted=train_df_polluted,
                                                                              test_df_polluted=test_df_polluted,
                                                                              error_type=error_type,
                                                                              ml_algorithm=ml_algorithm,
                                                                              ds_name=ds_name,
                                                                              pre_pollution_setting_id=pre_pollution_setting_id,
                                                                              iteration=iteration,
                                                                              database_engine=database_engine,
                                                                              features_importance=None,
                                                                              metadata=metadata, max_budget=BUDGET)

            if cleaning_result.get_feature() is not None:
                temp_df = cleaning_configs[cleaning_result.get_feature()]['data']
                temp_df.loc[temp_df['pollution_level'] == -0.01, 'real_f1'] = cleaning_result.get_f1_score()
                cleaning_configs[cleaning_result.get_feature()]['data'] = temp_df.copy()

                train_df_polluted = cleaning_result.get_cleaned_train_df()
                test_df_polluted = cleaning_result.get_cleaned_test_df()
                BUDGET -= cleaning_result.get_used_budget()
            else:
                print('Nothing to clean anymore.')
                break
            iteration += 1

        current_time = time.time()
        print(f'Elapsed time for pre-pollution {pre_pollution_setting_id}: {round((current_time - start_time), 2)} seconds')
        time_tracker[pre_pollution_setting_id] = round((current_time - start_time), 2)

    print('Elapsed times for all pre-pollution settings:', time_tracker)
    average_time = sum(time_tracker.values()) / len(time_tracker)
    print(f'Average time for {ds_name}, {ml_algorithm.__name__}, {error_type.__name__}: {average_time} seconds')
    print('Finished dynamic greedy for', ds_name, ml_algorithm.__name__, error_type.__name__)


if __name__ == "__main__":

    ml_algorithms = {'SupportVectorMachineExperiment': SupportVectorMachineExperiment,
                     'MultilayerPerceptronExperiment': MultilayerPerceptronExperiment,
                     'KNeighborsExperiment': KNeighborsExperiment,
                     'GradientBoostingExperiment': GradientBoostingExperiment,
                     'RandomForrestExperiment': RandomForrestExperiment,
                     'SGDClassifierExperimentLinearRegression': SGDClassifierExperimentLinearRegression,
                     'SGDClassifierExperimentLogisticRegression': SGDClassifierExperimentLogisticRegression,
                     'SGDClassifierExperimentSVM': SGDClassifierExperimentSVM}

    error_types = {'MissingValuesModifier': MissingValuesModifier,
                   'CategoricalShiftModifier': CategoricalShiftModifier,
                   'ScalingModifier': ScalingModifier,
                   'GaussianNoiseModifier': GaussianNoiseModifier}

    parser = argparse.ArgumentParser()
    parser.add_argument('--ml_algorithm', default='SupportVectorMachineExperiment', type=str, help='Set the ML algorithm to use for the experiment.')
    parser.add_argument('--error_type', default='MissingValuesModifier', type=str, help='Set the error type to use for the experiment.')
    parser.add_argument('--dataset', default='SouthGermanCredit.csv', type=str, help='Set the dataset to use for the experiment.')
    parser.add_argument('--budget', default=50, type=int, help='Set the available budget for the experiment.')
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

    database_engine = create_engine(database_url, echo=False, connect_args={'timeout': 1000})

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    main(chosen_ml_algorithm, chosen_error_type, ds_name, budget, metadata, database_engine, pre_pollution_setting_ids)
