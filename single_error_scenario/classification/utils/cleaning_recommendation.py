import os
import time
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


# Step 1: Define the Strategy Interface
class RecommendationStrategy(ABC):
    @abstractmethod
    def recommend_and_clean(self, **kwargs):
        pass

    @abstractmethod
    def get_sorted_features_to_clean(self, **kwargs):
        pass


def write_prediction_results_to_db(results_df: DataFrame, ds_name: str, feature: str, ml_algorithm, iteration: int, error_type, pre_pollution_setting_id: int, db_engine):
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

    def update_cleaning_results(self, cleaned_dfs: dict, f1_score: float, used_budget: int, feature: str):
        if self.f1_score < f1_score:
            self.cleaned_train_df = cleaned_dfs['train']['data'].copy()
            self.cleaned_test_df = cleaned_dfs['test']['data'].copy()
            self.cleaned_train_indexes = cleaned_dfs['train']['indexes']
            self.cleaned_test_indexes = cleaned_dfs['test']['indexes']
            self.f1_score = f1_score
            self.used_budget = used_budget
            self.feature = feature

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

    def set_feature(self, feature):
        self.feature = feature

    def set_f1_score(self, f1_score):
        self.f1_score = f1_score
    
    def write_result_to_db(self, iteration, ds_name, ml_algorithm_str, error_type_name, pre_pollution_setting_id, db_engine, current_f1_score, predicted_f1_score, number_of_cleaned_cells, directory_name='dynamic_greedy'):

        cleaning_results_df = DataFrame({'iteration': [iteration], 'ml_algorithm': [ml_algorithm_str], 'error_type': [error_type_name],
                            'pre_pollution_setting_id': [pre_pollution_setting_id], 'feature': [self.feature],
                            'used_budget': [self.used_budget], 'f1_score': [self.f1_score], 'predicted_f1_score': predicted_f1_score, 'number_of_cleaned_cells': [number_of_cleaned_cells]})

        if directory_name == 'comet_light':
            table_name = f'comet_light_cleaning_results_{ds_name}_{ml_algorithm_str}_{error_type_name}'
        else:
            table_name = f'cleaning_results_{ds_name}_{ml_algorithm_str}_{error_type_name}'
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


class CometRecommendationStrategy(RecommendationStrategy):

    def __init__(self, write_to_db: bool = False, cleaner=None):
        self.cleaning_buffer = {}
        self.prediction_error_history = {}
        self.write_to_db = write_to_db
        self.cleaner = cleaner
        self.cleaned_features = {}

    def get_cleaning_buffer(self, feature):
        if feature not in self.cleaning_buffer:
            return None
        else:
            return self.cleaning_buffer[feature]

    def remove_cleaning_buffer(self, feature):
        if feature in self.cleaning_buffer:
            del self.cleaning_buffer[feature]

    def set_cleaning_buffer(self, feature, cleaned_dfs):
        self.cleaning_buffer[feature] = cleaned_dfs

    def get_raw_prediction_history(self, feature):
        return self.prediction_error_history[feature]

    def get_mean_prediction_error_history(self, feature):
        if feature not in self.prediction_error_history:
            return 0.0
        else:
            return sum(self.prediction_error_history[feature]) / float(len(self.prediction_error_history[feature]))

    def get_log_decay_prediction_error_history(self, feature):
        return sum([x * 0.9 ** i for i, x in enumerate(self.prediction_error_history[feature])])

    def update_prediction_error_history(self, feature, new_error):
        if feature not in self.prediction_error_history:
            self.prediction_error_history[feature] = [new_error]
        else:
            self.prediction_error_history[feature].append(new_error)

    def check_if_feature_detected_as_clean_before(self, feature):
        return feature in self.cleaned_features

    def set_feature_as_cleaned(self, feature):
        self.cleaned_features[feature] = True

    def recommend_and_clean(self, **kwargs):
        cleaning_configs: Dict[str, Dict[DataFrame, List[int], List[int]]] = kwargs.get('cleaning_configs')
        train_df: DataFrame = kwargs.get('train_df')
        test_df: DataFrame = kwargs.get('test_df')
        train_df_polluted: DataFrame = kwargs.get('train_df_polluted')
        test_df_polluted: DataFrame = kwargs.get('test_df_polluted')
        features_importance = kwargs.get('features_importance')
        ml_algorithm = kwargs.get('ml_algorithm')
        max_budget: int = kwargs.get('max_budget')

        iteration: int = kwargs.get('iteration')
        error_type = kwargs.get('error_type')
        pre_pollution_setting_id: int = kwargs.get('pre_pollution_setting_id')
        ds_name: str = kwargs.get('ds_name')
        db_engine = kwargs.get('database_engine')
        metadata: Dict = kwargs.get('metadata')

        feature_scores = {}
        predicted_f1_scores = {}

        for feature in cleaning_configs:
            predicted_f1_scores[feature] = self.predict_f1_score(cleaning_configs, db_engine, ds_name, error_type, feature, iteration, ml_algorithm,
                                  pre_pollution_setting_id)
            feature_scores[feature] = self.calculate_score(cleaning_configs[feature]['data'].copy(), predicted_f1_scores[feature])

        # sort the features by score
        sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)

        cleaning_result = CleaningResult()
        used_budget = 0
        for feature in sorted_features:
            print(f'Cleaning feature: {feature}', 'iteration:', iteration)
            if self.check_if_feature_detected_as_clean_before(feature):
                continue
            # if self.get_cleaning_buffer(feature) is not None:
            #     cleaned_dfs = self.get_cleaning_buffer(feature)
            #     print('found feature', feature, 'in buffer')
            else:
                if used_budget == max_budget:
                    continue
                cleaned_dfs = self.cleaner.clean(feature, train_df_polluted=train_df_polluted, test_df_polluted=test_df_polluted,
                                   train_df=train_df, test_df=test_df, cleaning_config=cleaning_configs[feature],
                                   cleaning_step_size=0.01)

                if self._check_if_feature_is_clean(cleaned_dfs):
                    self.set_feature_as_cleaned(feature)
                    print('Feature', feature, 'is clean, skipping...')
                    continue
                if self.get_cleaning_buffer(feature) is None:
                    used_budget += 1

            np.random.seed(42)
            exp = ml_algorithm(cleaned_dfs['train']['data'].copy(), cleaned_dfs['test']['data'].copy(), metadata[ds_name],
                             error_type.get_classname(), pre_pollution_setting_id)
            results = exp.run('', 'scenario', explain=False)
            f1_score_after_cleaning = float(results[exp.name]['scoring']['macro avg']['f1-score'])
            self.update_prediction_error_history(feature, f1_score_after_cleaning - predicted_f1_scores[feature])
            cleaning_result.update_cleaning_results(cleaned_dfs, f1_score_after_cleaning, used_budget, feature)
            cleaning_configs[feature] = update_config_data(cleaning_configs[feature], f1_score_after_cleaning, predicted_f1_scores[feature])

            if f1_score_after_cleaning < self.get_current_f1_score(cleaning_configs, feature):
                self.set_cleaning_buffer(feature, cleaned_dfs)
                print('Feature', feature, 'is not good enough, adding to buffer...')
            else:
                self.remove_cleaning_buffer(feature)
                if self.write_to_db and cleaning_result.get_feature() is not None:
                    # get number of cleaned cells, but check if the list is not None
                    number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
                    cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(),
                                                        error_type.get_classname(), pre_pollution_setting_id, db_engine, self.get_current_f1_score(cleaning_configs, feature), predicted_f1_scores[feature], number_of_cleaned_cells)
                return cleaning_result
        self.remove_cleaning_buffer(cleaning_result.get_feature())
        cleaning_result.update_used_budget(used_budget)
        print('fallback for feature', cleaning_result.get_feature(), 'is used', 'used_budget', used_budget)
        if self.write_to_db and cleaning_result.get_feature() is not None:
            number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
            cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(), error_type.get_classname(), pre_pollution_setting_id, db_engine, self.get_current_f1_score(cleaning_configs, cleaning_result.get_feature()), 0.5, number_of_cleaned_cells)
        return cleaning_result

    def get_number_of_cleaned_cells(self, cleaning_result):
        number_of_cleaned_cells = 0
        if cleaning_result.get_cleaned_train_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_train_indexes())
        if cleaning_result.get_cleaned_test_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_test_indexes())
        return number_of_cleaned_cells

    def get_current_f1_score(self, cleaning_configs, feature):
        return cleaning_configs[feature]['data'][cleaning_configs[feature]['data']['pollution_level'] == 0.0][
            'real_f1'].values[0]

    def _check_if_feature_is_clean(self, cleaned_dfs):
        return cleaned_dfs['train']['indexes'] is None and cleaned_dfs['test']['indexes'] is None

    def calculate_score(self, current_df, predicted_f1_score):
        current_df = current_df[
            current_df['pollution_level'] == current_df['pollution_level'].min()]
        return predicted_f1_score - 1.0 * (
                current_df['upper_confidence_border'].values[0] - current_df['lower_confidence_border'].values[0])

    def predict_f1_score(self, cleaning_configs, db_engine, ds_name, error_type, feature, iteration, ml_algorithm,
                         pre_pollution_setting_id):
        feature_prediction_df = cleaning_configs[feature]['data']
        # filter row with minimum pollution level
        current_df = feature_prediction_df[
            feature_prediction_df['pollution_level'] == feature_prediction_df['pollution_level'].min()]
        predicted_f1 = current_df['predicted_f1'].values[0]
        prediction_error_adjustment = self.get_mean_prediction_error_history(feature)
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

    def get_cleaning_buffer(self, feature):
        if feature not in self.cleaning_buffer:
            return None
        else:
            return self.cleaning_buffer[feature]

    def remove_cleaning_buffer(self, feature):
        if feature in self.cleaning_buffer:
            del self.cleaning_buffer[feature]

    def set_cleaning_buffer(self, feature, cleaned_dfs):
        self.cleaning_buffer[feature] = cleaned_dfs

    def get_raw_prediction_history(self, feature):
        return self.prediction_error_history[feature]

    def get_mean_prediction_error_history(self, feature):
        if feature not in self.prediction_error_history:
            return 0.0
        else:
            return sum(self.prediction_error_history[feature]) / float(len(self.prediction_error_history[feature]))

    def get_log_decay_prediction_error_history(self, feature):
        return sum([x * 0.9 ** i for i, x in enumerate(self.prediction_error_history[feature])])

    def update_prediction_error_history(self, feature, new_error):
        if feature not in self.prediction_error_history:
            self.prediction_error_history[feature] = [new_error]
        else:
            self.prediction_error_history[feature].append(new_error)

    def check_if_feature_detected_as_clean_before(self, feature):
        return feature in self.cleaned_features

    def set_feature_as_cleaned(self, feature):
        self.cleaned_features[feature] = True

    def get_sorted_features_to_clean(self, **kwargs):
        cleaning_configs: Dict[str, Dict[DataFrame, List[int], List[int]]] = kwargs.get('cleaning_configs')
        ml_algorithm = kwargs.get('ml_algorithm')

        iteration: int = kwargs.get('iteration')
        error_type = kwargs.get('error_type')
        pre_pollution_setting_id: int = kwargs.get('pre_pollution_setting_id')
        ds_name: str = kwargs.get('ds_name')
        db_engine = kwargs.get('database_engine')

        feature_scores = {}
        predicted_f1_scores = {}

        for feature in cleaning_configs:
            predicted_f1_scores[feature] = self.predict_f1_score(cleaning_configs, db_engine, ds_name, error_type, feature, iteration, ml_algorithm,
                                  pre_pollution_setting_id)
            feature_scores[feature] = self.calculate_score(cleaning_configs[feature]['data'].copy(), predicted_f1_scores[feature])

        # sort the features by score
        sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)

        return sorted_features, predicted_f1_scores

    def recommend_and_clean(self, **kwargs):
        cleaning_configs: Dict[str, Dict[DataFrame, List[int], List[int]]] = kwargs.get('cleaning_configs')
        train_df: DataFrame = kwargs.get('train_df')
        test_df: DataFrame = kwargs.get('test_df')
        train_df_polluted: DataFrame = kwargs.get('train_df_polluted')
        test_df_polluted: DataFrame = kwargs.get('test_df_polluted')
        ml_algorithm = kwargs.get('ml_algorithm')
        max_budget: int = kwargs.get('max_budget')

        iteration: int = kwargs.get('iteration')
        error_type = kwargs.get('error_type')
        pre_pollution_setting_id: int = kwargs.get('pre_pollution_setting_id')
        ds_name: str = kwargs.get('ds_name')
        db_engine = kwargs.get('database_engine')
        metadata: Dict = kwargs.get('metadata')
        candidates = kwargs.get('candidates')
        predicted_f1_scores = kwargs.get('predicted_f1_scores_et')


        cleaning_result = CleaningResult()
        used_budget = 0
        for feature in candidates:
            print(f'Cleaning feature: {feature}', 'iteration:', iteration)
            if self.check_if_feature_detected_as_clean_before(feature):
                continue
            else:
                if used_budget == max_budget:
                    continue
                cleaned_dfs = self.cleaner.clean(feature, train_df_polluted=train_df_polluted, test_df_polluted=test_df_polluted,
                                   train_df=train_df, test_df=test_df, cleaning_config=cleaning_configs[feature],
                                   cleaning_step_size=0.01)

                if self._check_if_feature_is_clean(cleaned_dfs):
                    self.set_feature_as_cleaned(feature)
                    print('Feature', feature, 'is clean, skipping...')
                    continue
                if self.get_cleaning_buffer(feature) is None:
                    used_budget += 1

            np.random.seed(42)
            exp = ml_algorithm(cleaned_dfs['train']['data'].copy(), cleaned_dfs['test']['data'].copy(), metadata[ds_name],
                             error_type.get_classname(), pre_pollution_setting_id)
            results = exp.run('', 'scenario', explain=False)
            f1_score_after_cleaning = float(results[exp.name]['scoring']['macro avg']['f1-score'])
            self.update_prediction_error_history(feature, f1_score_after_cleaning - predicted_f1_scores[feature])
            cleaning_result.update_cleaning_results(cleaned_dfs, f1_score_after_cleaning, used_budget, feature)
            cleaning_configs[feature] = update_config_data(cleaning_configs[feature], f1_score_after_cleaning, predicted_f1_scores[feature])

            if f1_score_after_cleaning < self.get_current_f1_score(cleaning_configs, feature):
                self.set_cleaning_buffer(feature, cleaned_dfs)
                print('Feature', feature, 'is not good enough, adding to buffer...')
            else:
                self.remove_cleaning_buffer(feature)
                if self.write_to_db and cleaning_result.get_feature() is not None:
                    # get number of cleaned cells, but check if the list is not None
                    number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
                    cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(),
                                                        error_type.get_classname(), pre_pollution_setting_id, db_engine, self.get_current_f1_score(cleaning_configs, feature), predicted_f1_scores[feature], number_of_cleaned_cells, directory_name='comet_light')
                return cleaning_result
        self.remove_cleaning_buffer(cleaning_result.get_feature())
        cleaning_result.update_used_budget(used_budget)
        print('fallback for feature', cleaning_result.get_feature(), 'is used', 'used_budget', used_budget)
        if self.write_to_db and cleaning_result.get_feature() is not None:
            number_of_cleaned_cells = self.get_number_of_cleaned_cells(cleaning_result)
            cleaning_result.write_result_to_db(iteration, ds_name, ml_algorithm.get_classname(), error_type.get_classname(), pre_pollution_setting_id, db_engine, self.get_current_f1_score(cleaning_configs, cleaning_result.get_feature()), 0.5, number_of_cleaned_cells, directory_name='comet_light')
        return cleaning_result

    def get_number_of_cleaned_cells(self, cleaning_result):
        number_of_cleaned_cells = 0
        if cleaning_result.get_cleaned_train_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_train_indexes())
        if cleaning_result.get_cleaned_test_indexes() is not None:
            number_of_cleaned_cells += len(cleaning_result.get_cleaned_test_indexes())
        return number_of_cleaned_cells

    def get_current_f1_score(self, cleaning_configs, feature):
        return cleaning_configs[feature]['data'][cleaning_configs[feature]['data']['pollution_level'] == 0.0]['real_f1'].values[0]

    def _check_if_feature_is_clean(self, cleaned_dfs):
        return cleaned_dfs['train']['indexes'] is None and cleaned_dfs['test']['indexes'] is None

    def calculate_score(self, current_df, predicted_f1_score):
        current_df = current_df[
            current_df['pollution_level'] == current_df['pollution_level'].min()]
        return predicted_f1_score - 1.0 * (
                current_df['upper_confidence_border'].values[0] - current_df['lower_confidence_border'].values[0])

    def predict_f1_score(self, cleaning_configs, db_engine, ds_name, error_type, feature, iteration, ml_algorithm,
                         pre_pollution_setting_id):
        feature_prediction_df = cleaning_configs[feature]['data']
        # filter row with minimum pollution level
        current_df = feature_prediction_df[
            feature_prediction_df['pollution_level'] == feature_prediction_df['pollution_level'].min()]
        predicted_f1 = current_df['predicted_f1'].values[0]
        prediction_error_adjustment = self.get_mean_prediction_error_history(feature)
        adjusted_predicted_f1 = predicted_f1 + prediction_error_adjustment
        # write the adjusted predicted f1 to the dataframe, where pollution level is minimum
        feature_prediction_df.loc[feature_prediction_df['pollution_level'] == feature_prediction_df[
            'pollution_level'].min(), 'predicted_f1'] = adjusted_predicted_f1
        return adjusted_predicted_f1


class StrategyC(RecommendationStrategy):
    def recommend_and_clean(self, **kwargs):
        # Strategy C expects data_z
        data_z = kwargs.get('data_z')
        # Implement recommendation logic for strategy C using data_z
        return "Recommendations based on strategy C with data_z"


class Recommender:
    def __init__(self, strategy: RecommendationStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: RecommendationStrategy):
        self.strategy = strategy

    def get_sorted_features_to_clean(self, **kwargs):
        return self.strategy.get_sorted_features_to_clean(**kwargs)

    def recommend_and_clean(self, **kwargs):
        return self.strategy.recommend_and_clean(**kwargs)
