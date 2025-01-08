import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.exc import NoSuchTableError, OperationalError
from sqlalchemy.sql import select
from typing import Dict, List

from classification.utils.util import load_data_from_db
from config.definitions import ROOT_DIR


class RecommendationStrategy(ABC):
    @abstractmethod
    def recommend_and_clean(self, **kwargs):
        pass

    @abstractmethod
    def get_sorted_features_to_clean(self, **kwargs):
        pass

    @abstractmethod
    def check_if_feature_detected_as_clean_before(self, feature, error_type_name):
        pass

    @abstractmethod
    def set_feature_as_cleaned(self, feature, error_type_name):
        pass


def write_prediction_results_to_db(results_df: DataFrame, ds_name: str, feature: str, ml_algorithm, iteration: int, error_type, pre_pollution_setting_id: int, db_engine):
    return
    error_type_str = error_type.get_classname()
    ml_algorithm_str = ml_algorithm.get_classname()
    results_df_copy = results_df.copy()
    results_df_copy['iteration'] = iteration

    complete_results = load_data_from_db(
        f'prediction_results_{ds_name}_{ml_algorithm_str}_{error_type_str}_{feature}_{pre_pollution_setting_id}', db_engine)

    if complete_results is not None and 'iteration' in complete_results.columns:
        complete_results = complete_results[complete_results['iteration'] != iteration]

    complete_results = pd.concat([complete_results, results_df_copy])
    try:
        complete_results.to_sql(
            name=f'prediction_results_{ds_name}_{ml_algorithm_str}_{error_type_str}_{feature}_{pre_pollution_setting_id}',
            con=db_engine, if_exists='replace', index=False)
    except Exception as e:
        print(e)


class CleaningResult:
    def __init__(self):
        self.cleaned_train_df = None
        self.cleaned_test_df = None
        self.cleaned_train_indexes = None
        self.cleaned_test_indexes = None
        self.f1_score = -1.0
        self.used_budget = None
        self.feature = None
        self.error_type_name = None

    def update_cleaning_results(self, cleaned_dfs: dict, f1_score: float, used_budget: int, feature: str, error_type_name: str):
        if self.f1_score < f1_score:
            self.cleaned_train_df = cleaned_dfs['train']['data'].copy()
            self.cleaned_test_df = cleaned_dfs['test']['data'].copy()
            self.cleaned_train_indexes = cleaned_dfs['train']['indexes']
            self.cleaned_test_indexes = cleaned_dfs['test']['indexes']
            self.f1_score = f1_score
            self.used_budget = used_budget
            self.feature = feature
            self.error_type_name = error_type_name
    def forced_update_cleaning_results(self, cleaned_dfs: dict, f1_score: float, used_budget: int, feature: str, error_type_name: str):
        self.cleaned_train_df = cleaned_dfs['train']['data'].copy()
        self.cleaned_test_df = cleaned_dfs['test']['data'].copy()
        self.cleaned_train_indexes = cleaned_dfs['train']['indexes']
        self.cleaned_test_indexes = cleaned_dfs['test']['indexes']
        self.f1_score = f1_score
        self.used_budget = used_budget
        self.feature = feature
        self.error_type_name = error_type_name

    def update_used_budget(self, used_budget: int):
        self.used_budget = used_budget

    def get_feature(self):
        return self.feature

    def get_cleaned_train_df(self):
        return self.cleaned_train_df

    def get_cleaned_test_df(self):
        return self.cleaned_test_df

    def get_cleaned_train_indexes(self):
        return self.cleaned_train_indexes

    def get_cleaned_test_indexes(self):
        return self.cleaned_test_indexes

    def get_f1_score(self):
        return self.f1_score

    def get_used_budget(self):
        return self.used_budget

    def get_error_type_name(self):
        return self.error_type_name

    def set_feature(self, feature):
        self.feature = feature

    def set_f1_score(self, f1_score):
        self.f1_score = f1_score

    def set_error_type_name(self, error_type_name):
        self.error_type_name = error_type_name
    
    def write_result_to_db(self, iteration, ds_name, ml_algorithm_str, error_type_name, pre_pollution_setting_id, db_engine, current_f1_score, predicted_f1_score, number_of_cleaned_cells, directory_name='dynamic_greedy'):

        cleaning_results_df = DataFrame({'iteration': [iteration], 'ml_algorithm': [ml_algorithm_str], 'error_type': [error_type_name],
                            'pre_pollution_setting_id': [pre_pollution_setting_id], 'feature': [self.feature],
                            'used_budget': [self.used_budget], 'f1_score': [self.f1_score], 'predicted_f1_score': predicted_f1_score, 'number_of_cleaned_cells': [number_of_cleaned_cells]})

        if directory_name == 'comet_light':
            table_name = f'comet_light_cleaning_results_{ds_name}_{ml_algorithm_str}'
        else:
            table_name = f'cleaning_results_{ds_name}_{ml_algorithm_str}'  # For Multi error handling
        with open(f'{ROOT_DIR}/slurm/{directory_name}/RESULTS/{table_name}_{pre_pollution_setting_id}.csv', 'a') as f:
            # if file is empty then write header
            if os.stat(f'{ROOT_DIR}/slurm/{directory_name}/RESULTS/{table_name}_{pre_pollution_setting_id}.csv').st_size == 0:
                first_entry_df = DataFrame(
                    {'iteration': [0], 'ml_algorithm': [ml_algorithm_str], 'error_type': [error_type_name],
                     'pre_pollution_setting_id': [pre_pollution_setting_id], 'feature': [''],
                     'used_budget': [0], 'f1_score': [current_f1_score],
                     'predicted_f1_score': [current_f1_score],
                     'number_of_cleaned_cells': [0]})
                concatenated_df = pd.concat([first_entry_df, cleaning_results_df], ignore_index=True)
                concatenated_df.to_csv(f, header=True, index=False)
            else:
                cleaning_results_df.to_csv(f, header=False, index=False)


def update_config_data(config, f1_score_after_cleaning, predicted_f1_score):
    data_df = config['data']
    min_pollution_level = data_df['pollution_level'].min()
    data_df.loc[data_df['pollution_level'] == min_pollution_level, 'real_f1'] = f1_score_after_cleaning
    data_df.loc[data_df['pollution_level'] == min_pollution_level, 'predicted_f1'] = predicted_f1_score
    config['data'] = data_df.copy()
    return config


class CostFunction:
    def __init__(self):
        self.feature_state: dict = {}

    def initialize_feature(self, feature: str, error_type_name: str):
        add_state = False
        if feature not in self.feature_state:
            self.feature_state[feature] = {error_type_name: {'clean_count': 0}}
            add_state = True
        elif error_type_name not in self.feature_state[feature]:
            self.feature_state[feature][error_type_name] = {'clean_count': 0}
            add_state = True
        return add_state

    def remove_feature(self, feature: str, error_type_name: str):
        if feature in self.feature_state and error_type_name in self.feature_state[feature]:
            del self.feature_state[feature][error_type_name]

    def update_features_clean_count(self, feature: str, error_type_name: str, clean_increment=1):
        self.feature_state[feature][error_type_name]['clean_count'] += clean_increment

    def one_time_cost_function(self, feature: str, error_type_name: str, fitting_check: bool):
        added_state = self.initialize_feature(feature, error_type_name)
        if fitting_check and added_state:
            self.remove_feature(feature, error_type_name)
        if added_state:
            return 2
        return 0

    def constant_cost_function(self, feature: str, error_type_name: str):
        self.initialize_feature(feature, error_type_name)
        return 1

    def linear_cost_function(self, feature: str, error_type_name: str):
        self.initialize_feature(feature, error_type_name)
        used_budget = (self.feature_state[feature][error_type_name]['clean_count'] * 1) + 1
        return used_budget

    def exponential_cost_function(self, feature: str, error_type_name: str):
        self.initialize_feature(feature, error_type_name)
        used_budget = 2 ** self.feature_state[feature][error_type_name]['clean_count']
        return used_budget

    def calculate_used_budget(self, feature: str, error_type_name: str, fitting_check=False):
        if error_type_name == 'CategoricalShiftModifier':
            used_budget = self.constant_cost_function(feature, error_type_name)
        elif error_type_name == 'MissingValuesModifier':
            used_budget = self.one_time_cost_function(feature, error_type_name, fitting_check)
        elif error_type_name == 'GaussianNoiseModifier':
            used_budget = self.linear_cost_function(feature, error_type_name)
        else:  # error_type_name == 'ScalingModifier':
            used_budget = self.constant_cost_function(feature, error_type_name)
        if fitting_check is False:
            self.update_features_clean_count(feature, error_type_name)
        return used_budget


class CometRecommendationStrategy(RecommendationStrategy):

    def __init__(self, write_to_db: bool = False, cleaner=None):
        self.cleaning_buffer = {}
        self.prediction_error_history = {}
        self.write_to_db = write_to_db
        self.cleaner = cleaner
        self.cleaned_features = {}
        self.cost_function = CostFunction()

    def get_cleaning_buffer(self, feature: str, error_type_name: str):
        if feature not in self.cleaning_buffer:
            return None
        if error_type_name not in self.cleaning_buffer[feature]:
            return None
        else:
            return self.cleaning_buffer[feature][error_type_name]

    def remove_cleaning_buffer(self, feature: str, error_type_name: str):
        if feature in self.cleaning_buffer and error_type_name in self.cleaning_buffer[feature]:
            del self.cleaning_buffer[feature][error_type_name]

    def set_cleaning_buffer(self, feature: str, cleaned_dfs, error_type_name: str):
        if feature not in self.cleaning_buffer:
            self.cleaning_buffer[feature] = {error_type_name: cleaned_dfs}
        else:
            self.cleaning_buffer[feature][error_type_name] = cleaned_dfs

    def get_raw_prediction_history(self, feature: str, error_type_name: str):
        return self.prediction_error_history[feature][error_type_name]

    def get_mean_prediction_error_history(self, feature: str, error_type_name: str):
        if feature not in self.prediction_error_history:
            return 0.0
        if error_type_name not in self.prediction_error_history[feature]:
            return 0.0
        prediction_error_history = self.prediction_error_history[feature][error_type_name]
        return sum(prediction_error_history) / float(len(prediction_error_history))

    def get_log_decay_prediction_error_history(self, feature: str, error_type_name: str):  # unused
        return sum([x * 0.9 ** i for i, x in enumerate(self.prediction_error_history[feature][error_type_name])])

    def update_prediction_error_history(self, feature: str, error_type_name: str, new_error):
        if feature not in self.prediction_error_history:
            self.prediction_error_history[feature] = {error_type_name: [new_error]}
        elif error_type_name not in self.prediction_error_history[feature]:
            self.prediction_error_history[feature][error_type_name] = [new_error]
        else:
            self.prediction_error_history[feature][error_type_name].append(new_error)

    def check_if_feature_detected_as_clean_before(self, feature: str, error_type_name: str):
        return feature in self.cleaned_features and error_type_name in self.cleaned_features[feature]

    def set_feature_as_cleaned(self, feature: str, error_type_name: str):
        if feature not in self.cleaned_features:
            self.cleaned_features[feature] = [error_type_name]
        else:
            self.cleaned_features[feature].append(error_type_name)

    def recommend_and_clean(self, **kwargs):
        cleaning_configs_et: Dict[str, Dict[DataFrame, List[int], List[int]]] = kwargs.get('cleaning_configs')
        train_df: DataFrame = kwargs.get('train_df')
        test_df: DataFrame = kwargs.get('test_df')
        train_df_polluted: DataFrame = kwargs.get('train_df_polluted')
        test_df_polluted: DataFrame = kwargs.get('test_df_polluted')
        features_importance = kwargs.get('features_importance')
        ml_algorithm = kwargs.get('ml_algorithm')
        max_budget: int = kwargs.get('max_budget')

        iteration: int = kwargs.get('iteration')
        error_types: list = kwargs.get('error_types')
        pre_pollution_setting_id: int = kwargs.get('pre_pollution_setting_id')
        ds_name: str = kwargs.get('ds_name')
        db_engine = kwargs.get('database_engine')
        metadata: Dict = kwargs.get('metadata')
        error_map_train = kwargs.get('error_map_train')
        error_map_test = kwargs.get('error_map_test')

        feature_scores_et = {}
        predicted_f1_scores_et = {}

        for error_type in error_types:
            predicted_f1_scores_et[error_type.__name__] = {}
            feature_scores_et[error_type.__name__] = {}
            cleaning_configs = cleaning_configs_et[error_type.__name__]
            for feature in cleaning_configs:
                if self.get_cleaning_buffer(feature, error_type.__name__) is None:
                    additional_needed_budget = self.cost_function.calculate_used_budget(feature, error_type.__name__, fitting_check=True)
                else:
                    additional_needed_budget = 0
                predicted_f1_scores_et[error_type.__name__][feature] = self.predict_f1_score(cleaning_configs, db_engine,
                                                                                             ds_name, error_type, feature, iteration, ml_algorithm, pre_pollution_setting_id)
                feature_scores_et[error_type.__name__][feature] = self.calculate_score(cleaning_configs[feature]['data'].copy(), predicted_f1_scores_et[error_type.__name__][feature], additional_needed_budget)

        sorted_features: list = []
        for error_type in error_types:
            for feature in feature_scores_et[error_type.__name__]:
                sorted_features.append({'error_type': error_type.__name__, 'score': feature_scores_et[error_type.__name__][feature], 'feature': feature})
        # sort the tuples in sorted_features by score
        sorted_features = sorted(sorted_features, key=lambda x: x['score'], reverse=True)

        cleaning_result = CleaningResult()
        used_budget = 0
        for feature_tuple in sorted_features:
            feature = feature_tuple['feature']
            cleaning_configs = cleaning_configs_et[feature_tuple['error_type']]
            error_type = [error_type for error_type in error_types if error_type.__name__ == feature_tuple['error_type']][0]  # get the error type object

            print(f'Cleaning feature: {feature}', 'iteration:', iteration)
            if self.check_if_feature_detected_as_clean_before(feature, error_type.__name__):
                continue
            else:
                if ((used_budget == max_budget) or
                        (used_budget + self.cost_function.calculate_used_budget(feature, error_type.__name__, fitting_check=True) > max_budget and
                         self.get_cleaning_buffer(feature, error_type.__name__) is None)):
                    continue
                cleaned_dfs = self.cleaner.clean(feature, train_df_polluted=train_df_polluted, test_df_polluted=test_df_polluted,
                                   train_df=train_df, test_df=test_df, cleaning_config=cleaning_configs[feature],
                                   cleaning_step_size=0.01, error_map_train=error_map_train, error_map_test=error_map_test, error_type=error_type)

                if self._check_if_feature_is_clean(cleaned_dfs):
                    self.set_feature_as_cleaned(feature, error_type.__name__)
                    print('Feature', feature, 'is clean, skipping...')
                    continue
                if self.get_cleaning_buffer(feature, error_type.__name__) is None:
                    used_budget += self.cost_function.calculate_used_budget(feature, error_type.__name__)

            np.random.seed(42)
            exp = ml_algorithm(cleaned_dfs['train']['data'].copy(), cleaned_dfs['test']['data'].copy(), metadata[ds_name], 'multi_error', pre_pollution_setting_id)
            results = exp.run('', 'scenario', explain=False)
            f1_score_after_cleaning = float(results[exp.name]['scoring']['macro avg']['f1-score'])
            self.update_prediction_error_history(feature, error_type.__name__, f1_score_after_cleaning - predicted_f1_scores_et[error_type.__name__][feature])
            cleaning_result.update_cleaning_results(cleaned_dfs, f1_score_after_cleaning, used_budget, feature, error_type.__name__)
            cleaning_configs[feature] = update_config_data(cleaning_configs[feature], f1_score_after_cleaning, predicted_f1_scores_et[error_type.__name__][feature])

            if f1_score_after_cleaning < self.get_current_f1_score(cleaning_configs, feature):
                self.set_cleaning_buffer(feature, cleaned_dfs, error_type.__name__)
                print('Feature', feature, 'is not good enough, adding to buffer...')
            else:
                self.remove_cleaning_buffer(feature, error_type.__name__)
                if self.write_to_db and cleaning_result.get_feature() is not None:
                    # get number of cleaned cells, but check if the list is not None
                    number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
                    cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(),
                                                        error_type.get_classname(), pre_pollution_setting_id, db_engine, self.get_current_f1_score(cleaning_configs, feature), predicted_f1_scores_et[error_type.__name__][feature], number_of_cleaned_cells)
                return cleaning_result
        self.remove_cleaning_buffer(cleaning_result.get_feature(), cleaning_result.get_error_type_name())
        cleaning_result.update_used_budget(used_budget)
        print('fallback for feature', cleaning_result.get_feature(), 'is used', 'used_budget', used_budget)
        if self.write_to_db and cleaning_result.get_feature() is not None:
            number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
            cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(), cleaning_result.get_error_type_name(), pre_pollution_setting_id, db_engine, self.get_current_f1_score(cleaning_configs, list(cleaning_configs.keys())[0]), 0.5, number_of_cleaned_cells)
        return cleaning_result

    def clean_by_candidates(self, **kwargs):
        cleaning_configs_et: Dict[str, Dict[DataFrame, List[int], List[int]]] = kwargs.get('cleaning_configs')
        train_df: DataFrame = kwargs.get('train_df')
        test_df: DataFrame = kwargs.get('test_df')
        train_df_polluted: DataFrame = kwargs.get('train_df_polluted')
        test_df_polluted: DataFrame = kwargs.get('test_df_polluted')
        ml_algorithm = kwargs.get('ml_algorithm')
        max_budget: int = kwargs.get('max_budget')

        iteration: int = kwargs.get('iteration')
        error_types: list = kwargs.get('error_types')
        pre_pollution_setting_id: int = kwargs.get('pre_pollution_setting_id')
        ds_name: str = kwargs.get('ds_name')
        db_engine = kwargs.get('database_engine')
        metadata: Dict = kwargs.get('metadata')
        error_map_train = kwargs.get('error_map_train')
        error_map_test = kwargs.get('error_map_test')
        candidates = kwargs.get('candidates')
        predicted_f1_scores_et = kwargs.get('predicted_f1_scores_et')

        cleaning_result = CleaningResult()
        used_budget = 0
        for feature_tuple in candidates:
            feature = feature_tuple['feature']
            cleaning_configs = cleaning_configs_et[feature_tuple['error_type']]
            error_type = \
            [error_type for error_type in error_types if error_type.__name__ == feature_tuple['error_type']][0]  # get the error type object

            print(f'Cleaning feature: {feature}', 'iteration:', iteration)
            if self.check_if_feature_detected_as_clean_before(feature, error_type.__name__):
                continue
            else:
                if ((used_budget == max_budget) or
                        (used_budget + self.cost_function.calculate_used_budget(feature, error_type.__name__,
                                                                                fitting_check=True) > max_budget and
                         self.get_cleaning_buffer(feature, error_type.__name__) is None)):
                    continue
                cleaned_dfs = self.cleaner.clean(feature, train_df_polluted=train_df_polluted,
                                                 test_df_polluted=test_df_polluted,
                                                 train_df=train_df, test_df=test_df,
                                                 cleaning_config=cleaning_configs[feature],
                                                 cleaning_step_size=0.01, error_map_train=error_map_train,
                                                 error_map_test=error_map_test, error_type=error_type)

                if self._check_if_feature_is_clean(cleaned_dfs):
                    self.set_feature_as_cleaned(feature, error_type.__name__)
                    print('Feature', feature, 'is clean, skipping...')
                    continue
                if self.get_cleaning_buffer(feature, error_type.__name__) is None:
                    used_budget += self.cost_function.calculate_used_budget(feature, error_type.__name__)

            np.random.seed(42)
            exp = ml_algorithm(cleaned_dfs['train']['data'].copy(), cleaned_dfs['test']['data'].copy(),
                               metadata[ds_name], 'multi_error', pre_pollution_setting_id)
            results = exp.run('', 'scenario', explain=False)
            f1_score_after_cleaning = float(results[exp.name]['scoring']['macro avg']['f1-score'])
            self.update_prediction_error_history(feature, error_type.__name__,
                                                 f1_score_after_cleaning - predicted_f1_scores_et[error_type.__name__][
                                                     feature])
            cleaning_result.update_cleaning_results(cleaned_dfs, f1_score_after_cleaning, used_budget, feature,
                                                    error_type.__name__)
            cleaning_configs[feature] = update_config_data(cleaning_configs[feature], f1_score_after_cleaning,
                                                           predicted_f1_scores_et[error_type.__name__][feature])

            if f1_score_after_cleaning < self.get_current_f1_score(cleaning_configs, feature):
                self.set_cleaning_buffer(feature, cleaned_dfs, error_type.__name__)
                print('Feature', feature, 'is not good enough, adding to buffer...')
            else:
                self.remove_cleaning_buffer(feature, error_type.__name__)
                if self.write_to_db and cleaning_result.get_feature() is not None:
                    # get number of cleaned cells, but check if the list is not None
                    number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
                    cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(),
                                                       error_type.get_classname(), pre_pollution_setting_id, db_engine,
                                                       self.get_current_f1_score(cleaning_configs, feature),
                                                       predicted_f1_scores_et[error_type.__name__][feature],
                                                       number_of_cleaned_cells)
                return cleaning_result
        self.remove_cleaning_buffer(cleaning_result.get_feature(), cleaning_result.get_error_type_name())
        cleaning_result.update_used_budget(used_budget)
        print('fallback for feature', cleaning_result.get_feature(), 'is used', 'used_budget', used_budget)
        if self.write_to_db and cleaning_result.get_feature() is not None:
            number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
            cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(),
                                               cleaning_result.get_error_type_name(), pre_pollution_setting_id,
                                               db_engine, self.get_current_f1_score(cleaning_configs,
                                                                                    list(cleaning_configs.keys())[0]),
                                               0.5, number_of_cleaned_cells)
        return cleaning_result

    def get_sorted_features_to_clean(self, **kwargs):
        return None


    def get_number_of_cleaned_cells(self, cleaning_result):
        number_of_cleaned_cells = 0
        if cleaning_result.get_cleaned_train_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_train_indexes())
        if cleaning_result.get_cleaned_test_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_test_indexes())
        return number_of_cleaned_cells

    def get_current_f1_score(self, cleaning_configs, feature: str):
        return cleaning_configs[feature]['data'][cleaning_configs[feature]['data']['pollution_level'] == 0.0]['real_f1'].values[0]

    def _check_if_feature_is_clean(self, cleaned_dfs):
        return cleaned_dfs['train']['indexes'] is None and cleaned_dfs['test']['indexes'] is None

    def calculate_score(self, current_df, predicted_f1_score, additional_needed_budget):
        current_df = current_df[
            current_df['pollution_level'] == current_df['pollution_level'].min()]
        if additional_needed_budget == 0:
            additional_needed_budget = 1  # set it to 1 to not prioritize feature-error combinations from the buffer
        return (predicted_f1_score - 1.0 * (
                current_df['upper_confidence_border'].values[0] - current_df['lower_confidence_border'].values[0]))/additional_needed_budget  # divide calculated score by additional needed budget caused by this cleaning operation

    def predict_f1_score(self, cleaning_configs, db_engine, ds_name, error_type, feature, iteration, ml_algorithm,
                         pre_pollution_setting_id):
        feature_prediction_df = cleaning_configs[feature]['data']
        # filter row with minimum pollution level
        current_df = feature_prediction_df[
            feature_prediction_df['pollution_level'] == feature_prediction_df['pollution_level'].min()]
        predicted_f1 = current_df['predicted_f1'].values[0]
        prediction_error_adjustment = self.get_mean_prediction_error_history(feature, error_type.__name__)
        adjusted_predicted_f1 = predicted_f1 + prediction_error_adjustment
        # write the adjusted predicted f1 to the dataframe, where pollution level is minimum
        feature_prediction_df.loc[feature_prediction_df['pollution_level'] == feature_prediction_df[
            'pollution_level'].min(), 'predicted_f1'] = adjusted_predicted_f1
        if self.write_to_db:
            write_prediction_results_to_db(feature_prediction_df, ds_name, feature, ml_algorithm, iteration, error_type,
                                           pre_pollution_setting_id, db_engine)
        return adjusted_predicted_f1


class CometLightRecommendationStrategy(RecommendationStrategy):

    def __init__(self, write_to_db: bool = False, cleaner=None):
        self.cleaning_buffer = {}
        self.prediction_error_history = {}
        self.write_to_db = write_to_db
        self.cleaner = cleaner
        self.cleaned_features = {}
        self.cost_function = CostFunction()

    def get_cleaning_buffer(self, feature: str, error_type_name: str):
        if feature not in self.cleaning_buffer:
            return None
        if error_type_name not in self.cleaning_buffer[feature]:
            return None
        else:
            return self.cleaning_buffer[feature][error_type_name]

    def remove_cleaning_buffer(self, feature: str, error_type_name: str):
        if feature in self.cleaning_buffer and error_type_name in self.cleaning_buffer[feature]:
            del self.cleaning_buffer[feature][error_type_name]

    def set_cleaning_buffer(self, feature: str, cleaned_dfs, error_type_name: str):
        if feature not in self.cleaning_buffer:
            self.cleaning_buffer[feature] = {error_type_name: cleaned_dfs}
        else:
            self.cleaning_buffer[feature][error_type_name] = cleaned_dfs

    def get_raw_prediction_history(self, feature: str, error_type_name: str):
        return self.prediction_error_history[feature][error_type_name]

    def get_mean_prediction_error_history(self, feature: str, error_type_name: str):
        if feature not in self.prediction_error_history:
            return 0.0
        if error_type_name not in self.prediction_error_history[feature]:
            return 0.0
        prediction_error_history = self.prediction_error_history[feature][error_type_name]
        return sum(prediction_error_history) / float(len(prediction_error_history))

    def get_log_decay_prediction_error_history(self, feature: str, error_type_name: str):  # unused
        return sum([x * 0.9 ** i for i, x in enumerate(self.prediction_error_history[feature][error_type_name])])

    def update_prediction_error_history(self, feature: str, error_type_name: str, new_error):
        if feature not in self.prediction_error_history:
            self.prediction_error_history[feature] = {error_type_name: [new_error]}
        elif error_type_name not in self.prediction_error_history[feature]:
            self.prediction_error_history[feature][error_type_name] = [new_error]
        else:
            self.prediction_error_history[feature][error_type_name].append(new_error)

    def check_if_feature_detected_as_clean_before(self, feature: str, error_type_name: str):
        return feature in self.cleaned_features and error_type_name in self.cleaned_features[feature]

    def set_feature_as_cleaned(self, feature: str, error_type_name: str):
        if feature not in self.cleaned_features:
            self.cleaned_features[feature] = [error_type_name]
        else:
            self.cleaned_features[feature].append(error_type_name)

    def recommend_and_clean(self, **kwargs):
        cleaning_configs_et: Dict[str, Dict[DataFrame, List[int], List[int]]] = kwargs.get('cleaning_configs')
        train_df: DataFrame = kwargs.get('train_df')
        test_df: DataFrame = kwargs.get('test_df')
        train_df_polluted: DataFrame = kwargs.get('train_df_polluted')
        test_df_polluted: DataFrame = kwargs.get('test_df_polluted')
        ml_algorithm = kwargs.get('ml_algorithm')
        max_budget: int = kwargs.get('max_budget')

        iteration: int = kwargs.get('iteration')
        error_types: list = kwargs.get('error_types')
        pre_pollution_setting_id: int = kwargs.get('pre_pollution_setting_id')
        ds_name: str = kwargs.get('ds_name')
        db_engine = kwargs.get('database_engine')
        metadata: Dict = kwargs.get('metadata')
        error_map_train = kwargs.get('error_map_train')
        error_map_test = kwargs.get('error_map_test')
        candidates = kwargs.get('candidates')
        predicted_f1_scores_et = kwargs.get('predicted_f1_scores_et')

        cleaning_result = CleaningResult()
        used_budget = 0
        for feature_tuple in candidates:
            feature = feature_tuple['feature']
            cleaning_configs = cleaning_configs_et[feature_tuple['error_type']]
            error_type = [error_type for error_type in error_types if error_type.__name__ == feature_tuple['error_type']][0]  # get the error type object

            print(f'Cleaning feature: {feature}', 'iteration:', iteration)
            if self.check_if_feature_detected_as_clean_before(feature, error_type.__name__):
                continue
            else:
                if ((used_budget == max_budget) or
                        (used_budget + self.cost_function.calculate_used_budget(feature, error_type.__name__,
                                                                                fitting_check=True) > max_budget and
                         self.get_cleaning_buffer(feature, error_type.__name__) is None)):
                    continue
                cleaned_dfs = self.cleaner.clean(feature, train_df_polluted=train_df_polluted,
                                                 test_df_polluted=test_df_polluted,
                                                 train_df=train_df, test_df=test_df,
                                                 cleaning_config=cleaning_configs[feature],
                                                 cleaning_step_size=0.01, error_map_train=error_map_train,
                                                 error_map_test=error_map_test, error_type=error_type)

                if self._check_if_feature_is_clean(cleaned_dfs):
                    self.set_feature_as_cleaned(feature, error_type.__name__)
                    print('Feature', feature, 'is clean, skipping...')
                    continue
                if self.get_cleaning_buffer(feature, error_type.__name__) is None:
                    used_budget += self.cost_function.calculate_used_budget(feature, error_type.__name__)

            np.random.seed(42)
            exp = ml_algorithm(cleaned_dfs['train']['data'].copy(), cleaned_dfs['test']['data'].copy(),
                               metadata[ds_name], 'multi_error', pre_pollution_setting_id)
            results = exp.run('', 'scenario', explain=False)
            f1_score_after_cleaning = float(results[exp.name]['scoring']['macro avg']['f1-score'])
            self.update_prediction_error_history(feature, error_type.__name__,
                                                 f1_score_after_cleaning - predicted_f1_scores_et[error_type.__name__][
                                                     feature])
            cleaning_result.update_cleaning_results(cleaned_dfs, f1_score_after_cleaning, used_budget, feature,
                                                    error_type.__name__)
            cleaning_configs[feature] = update_config_data(cleaning_configs[feature], f1_score_after_cleaning,
                                                           predicted_f1_scores_et[error_type.__name__][feature])

            if f1_score_after_cleaning < self.get_current_f1_score(cleaning_configs, feature):
                self.set_cleaning_buffer(feature, cleaned_dfs, error_type.__name__)
                print('Feature', feature, 'is not good enough, adding to buffer...')
            else:
                self.remove_cleaning_buffer(feature, error_type.__name__)
                if self.write_to_db and cleaning_result.get_feature() is not None:
                    # get number of cleaned cells, but check if the list is not None
                    number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
                    cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(),
                                                       error_type.get_classname(), pre_pollution_setting_id, db_engine,
                                                       self.get_current_f1_score(cleaning_configs, feature),
                                                       predicted_f1_scores_et[error_type.__name__][feature],
                                                       number_of_cleaned_cells, directory_name='comet_light')
                return cleaning_result
        self.remove_cleaning_buffer(cleaning_result.get_feature(), cleaning_result.get_error_type_name())
        cleaning_result.update_used_budget(used_budget)
        print('fallback for feature', cleaning_result.get_feature(), 'is used', 'used_budget', used_budget)
        if self.write_to_db and cleaning_result.get_feature() is not None:
            number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
            cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(),
                                               cleaning_result.get_error_type_name(), pre_pollution_setting_id,
                                               db_engine, self.get_current_f1_score(cleaning_configs,
                                                                                    list(cleaning_configs.keys())[0]),
                                               0.5, number_of_cleaned_cells, directory_name='comet_light')
        return cleaning_result

    def get_sorted_features_to_clean(self, **kwargs):
        cleaning_configs_et: Dict[str, Dict[DataFrame, List[int], List[int]]] = kwargs.get('cleaning_configs')
        ml_algorithm = kwargs.get('ml_algorithm')

        iteration: int = kwargs.get('iteration')
        error_types: list = kwargs.get('error_types')
        pre_pollution_setting_id: int = kwargs.get('pre_pollution_setting_id')
        ds_name: str = kwargs.get('ds_name')
        db_engine = kwargs.get('database_engine')

        feature_scores_et = {}
        predicted_f1_scores_et = {}

        for error_type in error_types:
            predicted_f1_scores_et[error_type.__name__] = {}
            feature_scores_et[error_type.__name__] = {}
            cleaning_configs = cleaning_configs_et[error_type.__name__]
            for feature in cleaning_configs:
                if self.get_cleaning_buffer(feature, error_type.__name__) is None:
                    additional_needed_budget = self.cost_function.calculate_used_budget(feature, error_type.__name__, fitting_check=True)
                else:
                    additional_needed_budget = 0
                predicted_f1_scores_et[error_type.__name__][feature] = self.predict_f1_score(cleaning_configs, db_engine,
                                                                                             ds_name, error_type, feature, iteration, ml_algorithm, pre_pollution_setting_id)
                feature_scores_et[error_type.__name__][feature] = self.calculate_score(cleaning_configs[feature]['data'].copy(), predicted_f1_scores_et[error_type.__name__][feature], additional_needed_budget)

        sorted_features: list = []
        for error_type in error_types:
            for feature in feature_scores_et[error_type.__name__]:
                sorted_features.append({'error_type': error_type.__name__, 'score': feature_scores_et[error_type.__name__][feature], 'feature': feature})
        # sort the tuples in sorted_features by score
        sorted_features = sorted(sorted_features, key=lambda x: x['score'], reverse=True)
        return sorted_features, predicted_f1_scores_et


    def get_number_of_cleaned_cells(self, cleaning_result):
        number_of_cleaned_cells = 0
        if cleaning_result.get_cleaned_train_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_train_indexes())
        if cleaning_result.get_cleaned_test_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_test_indexes())
        return number_of_cleaned_cells

    def get_current_f1_score(self, cleaning_configs, feature: str):
        return cleaning_configs[feature]['data'][cleaning_configs[feature]['data']['pollution_level'] == 0.0]['real_f1'].values[0]

    def _check_if_feature_is_clean(self, cleaned_dfs):
        return cleaned_dfs['train']['indexes'] is None and cleaned_dfs['test']['indexes'] is None

    def calculate_score(self, current_df, predicted_f1_score, additional_needed_budget):
        current_df = current_df[
            current_df['pollution_level'] == current_df['pollution_level'].min()]
        if additional_needed_budget == 0:
            additional_needed_budget = 1  # set it to 1 to not prioritize feature-error combinations from the buffer
        return (predicted_f1_score - 1.0 * (
                current_df['upper_confidence_border'].values[0] - current_df['lower_confidence_border'].values[0]))/additional_needed_budget  # divide calculated score by additional needed budget caused by this cleaning operation

    def predict_f1_score(self, cleaning_configs, db_engine, ds_name, error_type, feature, iteration, ml_algorithm,
                         pre_pollution_setting_id):
        feature_prediction_df = cleaning_configs[feature]['data']
        # filter row with minimum pollution level
        current_df = feature_prediction_df[
            feature_prediction_df['pollution_level'] == feature_prediction_df['pollution_level'].min()]
        predicted_f1 = current_df['predicted_f1'].values[0]
        prediction_error_adjustment = self.get_mean_prediction_error_history(feature, error_type.__name__)
        adjusted_predicted_f1 = predicted_f1 + prediction_error_adjustment
        # write the adjusted predicted f1 to the dataframe, where pollution level is minimum
        feature_prediction_df.loc[feature_prediction_df['pollution_level'] == feature_prediction_df['pollution_level'].min(), 'predicted_f1'] = adjusted_predicted_f1
        if self.write_to_db:
            write_prediction_results_to_db(feature_prediction_df, ds_name, feature, ml_algorithm, iteration, error_type,
                                           pre_pollution_setting_id, db_engine)
        return adjusted_predicted_f1


class StrategyC(RecommendationStrategy):
    def recommend_and_clean(self, **kwargs):
        data_z = kwargs.get('data_z')
        return "Recommendations based on strategy C with data_z"

    def check_if_feature_detected_as_clean_before(self, feature: str, error_type_name: str):
        return "Check if feature detected as clean before based on strategy C with data_z"

    def set_feature_as_cleaned(self, feature: str, error_type_name: str):
        return "Set feature as cleaned based on strategy C with data_z"


class Recommender:
    def __init__(self, strategy: RecommendationStrategy):
        self.cost_function = CostFunction()
        self.strategy = strategy

    def set_strategy(self, strategy: RecommendationStrategy):
        self.strategy = strategy

    def recommend_and_clean(self, **kwargs):
        return self.strategy.recommend_and_clean(**kwargs)

    def get_sorted_features_to_clean(self, **kwargs):
        return self.strategy.get_sorted_features_to_clean(**kwargs)

    def check_if_feature_detected_as_clean_before(self, feature: str, error_type_name: str):
        return self.strategy.check_if_feature_detected_as_clean_before(feature, error_type_name)

    def set_feature_as_cleaned(self, feature: str, error_type_name: str):
        return self.strategy.set_feature_as_cleaned(feature, error_type_name)

