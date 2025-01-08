from classification.utils.regression_model import *
from pandas import DataFrame
import numpy as np
from typing import Dict


def calculate_gain(row, original_f1_score: float):
    return row['predicted_poly_reg_f1'] - original_f1_score


class ArtificialPollution2:
    def __init__(self, metadata, database_url, pre_pollution_setting_id, modifier, pre_pollution_df, ds_name, pollution_setting):
        self.metadata: Dict = metadata
        self.DATABASE_URL: str = database_url
        self.pre_pollution_setting_id = pre_pollution_setting_id
        self.modifier = modifier
        self.ds_name: str = ds_name
        self.pollution_setting = pollution_setting
        filtered_history_df, is_empty = self.get_filtered_history_df(pre_pollution_df, 277712)
        self.train_df, self.test_df = self.get_clean_training_and_test_df(filtered_history_df, ds_name)
        self.train_df_polluted, self.test_df_polluted = self.get_current_polluted_training_and_test_df(filtered_history_df,
                                                                                             ds_name,
                                                                                             pollution_setting)

    def artificial_pollution(self, train_df_polluted, test_df_polluted, ds_name: str):
        feature_candidates_for_cleaning = self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols']
        feature_candidates_for_cleaning = self.filter_features_on_type(feature_candidates_for_cleaning, self.metadata, ds_name)

        feature_candidates_for_cleaning_temp = []
        for feature in feature_candidates_for_cleaning:
            if feature in self.pollution_setting['train'] and feature in self.pollution_setting['test']:
                if self.pollution_setting['train'][feature] != 0.0 and self.pollution_setting['test'][feature] != 0.0:
                    feature_candidates_for_cleaning_temp.append(feature)
        feature_candidates_for_cleaning = feature_candidates_for_cleaning_temp

        pollution_results = {}
        for feature in feature_candidates_for_cleaning:
            pollution_configs = self.execute_pollution(feature, train_df_polluted, test_df_polluted)
            pollution_results[feature] = pollution_configs
        return pollution_results

    def filter_features_on_type(self, feature_candidates_for_cleaning, metadata, ds_name: str):
        feature_candidates_for_cleaning_temp = []
        for feature in feature_candidates_for_cleaning:
            if feature in metadata[ds_name]['categorical_cols']:
                feature_type = 'categorical_col'
            else:
                feature_type = 'numerical_col'
            if feature_type == self.modifier.restricted_to or self.modifier.restricted_to == '':
                feature_candidates_for_cleaning_temp.append(feature)
        return feature_candidates_for_cleaning_temp

    def get_filtered_history_df(self, history_df: DataFrame, random_seed: int):
        np.random.seed(random_seed)
        filtered_history_df = history_df.copy()
        filtered_history_df['seed'] = filtered_history_df['seed'].astype(int)
        filtered_history_df['pollution_level'] = filtered_history_df['pollution_level'].astype(float)
        filtered_history_df = filtered_history_df[filtered_history_df['seed'] == random_seed]
        filtered_history_df = filtered_history_df[filtered_history_df['polluter'] == self.modifier.__name__]
        is_empty = filtered_history_df.empty
        return filtered_history_df, is_empty

    def get_clean_training_and_test_df(self, filtered_history_df: DataFrame, ds_name: str):
        df_clean = filtered_history_df[filtered_history_df['pollution_level'] == 0.0]
        train_df = df_clean[df_clean['train/test'] == 'train']
        test_df = df_clean[df_clean['train/test'] == 'test']
        train_df = train_df.reset_index(drop=True)
        train_df = train_df[self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [self.metadata[ds_name]['target']]]
        test_df = test_df.reset_index(drop=True)
        test_df = test_df[self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [self.metadata[ds_name]['target']]]
        return train_df, test_df

    def get_current_polluted_training_and_test_df(self, filtered_history_df: DataFrame, ds_name: str, feature_wise_pollution_level: dict):
        df_pre_polluted = pd.DataFrame()
        for current_feature in self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [self.metadata[ds_name]['target']]:
            if current_feature in feature_wise_pollution_level['train']:
                current_pollution_level = feature_wise_pollution_level['train'][current_feature]
            else:
                current_pollution_level = 0.0
            filtered_temp_history_df = filtered_history_df[
                filtered_history_df['pollution_level'] == current_pollution_level].reset_index(drop=True)
            filtered_temp_history_df = filtered_temp_history_df[filtered_temp_history_df['train/test'] == 'train']
            filtered_temp_history_df = filtered_temp_history_df[[current_feature]]
            filtered_temp_history_df = filtered_temp_history_df.reset_index(drop=True)
            df_pre_polluted = df_pre_polluted.reset_index(drop=True)
            df_pre_polluted[current_feature] = filtered_temp_history_df[current_feature]

        train_df_pre_polluted = df_pre_polluted.copy()

        df_pre_polluted = pd.DataFrame()
        for current_feature in self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [self.metadata[ds_name]['target']]:
            if current_feature in feature_wise_pollution_level['test']:
                current_pollution_level = feature_wise_pollution_level['test'][current_feature]
            else:
                current_pollution_level = 0.0
            filtered_temp_history_df = filtered_history_df[
                filtered_history_df['pollution_level'] == current_pollution_level].reset_index(drop=True)
            filtered_temp_history_df = filtered_temp_history_df[filtered_temp_history_df['train/test'] == 'test']
            filtered_temp_history_df = filtered_temp_history_df[[current_feature]]
            filtered_temp_history_df = filtered_temp_history_df.reset_index(drop=True)
            df_pre_polluted = df_pre_polluted.reset_index(drop=True)
            df_pre_polluted[current_feature] = filtered_temp_history_df[current_feature]

        test_df_pre_polluted = df_pre_polluted.copy()

        train_df_pre_polluted = train_df_pre_polluted[self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [
            self.metadata[ds_name]['target']]]
        test_df_pre_polluted = test_df_pre_polluted[self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [
            self.metadata[ds_name]['target']]]
        return train_df_pre_polluted, test_df_pre_polluted

    def execute_pollution(self, feature: str, train_df_polluted, test_df_polluted):

        pollution_configs = []
        for random_seed in [87263, 53219, 78604, 2023, 38472, 11, 9834, 4567, 909090, 56789]:
            np.random.seed(random_seed)
            corrupted_train_df = train_df_polluted.copy()
            corrupted_test_df = test_df_polluted.copy()
            min_pollution_level = 0.01
            max_pollution_level = 0.02
            sample_size_train_df = int(len(train_df_polluted) * max_pollution_level)
            sampling_indexes_train = train_df_polluted.sample(n=sample_size_train_df, random_state=random_seed).index.values.tolist()
            sample_indexes_train_pre_pollution_step = {0.0: [], min_pollution_level: sampling_indexes_train[:int(len(sampling_indexes_train) / 2)], max_pollution_level: sampling_indexes_train[int(len(sampling_indexes_train) / 2):]}

            sample_size_test_df = int(len(test_df_polluted) * max_pollution_level)
            sampling_indexes_test = test_df_polluted.sample(n=sample_size_test_df, random_state=random_seed).index.values.tolist()
            sample_indexes_test_pre_pollution_step = {0.0: [], min_pollution_level: sampling_indexes_test[:int(len(sampling_indexes_test) / 2)], max_pollution_level: sampling_indexes_test[int(len(sampling_indexes_test) / 2):]}

            #pollution_configs.append({'pollution_level': round(0.0, 2), 'random_seed': random_seed, 'train_df': corrupted_train_df.copy(), 'test_df': corrupted_test_df.copy(), 'sampling_indexes_train': [], 'sampling_indexes_test': []})
            pollution_configs.append(PollutedFeature(feature, round(0.0, 2), random_seed, corrupted_train_df.copy(), corrupted_test_df.copy(), [], []))
            for pollution_level in [min_pollution_level, max_pollution_level]:
                pollution_level = round(pollution_level, 2)
                pollution_level_setting = {feature: 1.0}

                indexes_to_sample_train = sample_indexes_train_pre_pollution_step[pollution_level]
                corrupted_train_df_sample = train_df_polluted.iloc[indexes_to_sample_train]
                corrupted_train_df_sample = self.modifier(pollution_level_setting, corrupted_train_df, corrupted_train_df_sample, self.metadata[self.ds_name]).pollute2()
                corrupted_train_df.loc[indexes_to_sample_train, feature] = corrupted_train_df_sample[feature]

                indexes_to_sample_test = sample_indexes_test_pre_pollution_step[pollution_level]
                corrupted_test_df_sample = test_df_polluted.iloc[indexes_to_sample_test]
                corrupted_test_df_sample = self.modifier(pollution_level_setting, corrupted_test_df, corrupted_test_df_sample, self.metadata[self.ds_name]).pollute2()
                corrupted_test_df.loc[indexes_to_sample_test, feature] = corrupted_test_df_sample[feature]

                corrupted_train_df_copy = corrupted_train_df.copy()
                corrupted_test_df_copy = corrupted_test_df.copy()
                #pollution_configs.append({'pollution_level': round(pollution_level, 2), 'random_seed': random_seed, 'train_df': corrupted_train_df_copy, 'test_df': corrupted_test_df_copy, 'sampling_indexes_train': indexes_to_sample_train, 'sampling_indexes_test': indexes_to_sample_test})
                pollution_configs.append(PollutedFeature(feature, round(pollution_level, 2), random_seed, corrupted_train_df_copy, corrupted_test_df_copy, indexes_to_sample_train, indexes_to_sample_test))
        return pollution_configs


class PollutedFeature:
    def __init__(self, feature, pollution_level, random_seed, train_df, test_df, sampling_indexes_train, sampling_indexes_test):
        self.feature = feature
        self.pollution_level = pollution_level
        self.random_seed = random_seed
        self.train_df = train_df
        self.test_df = test_df
        self.sampling_indexes_train = sampling_indexes_train
        self.sampling_indexes_test = sampling_indexes_test

    def get_feature(self):
        return self.feature

    def get_pollution_level(self):
        return self.pollution_level

    def get_random_seed(self):
        return self.random_seed

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df

    def get_sampling_indexes_train(self):
        return self.sampling_indexes_train

    def get_sampling_indexes_test(self):
        return self.sampling_indexes_test
