from joblib import Parallel, delayed

from classification.utils.regression_model import *
from classification.utils.util import load_data_from_db
from sqlalchemy import create_engine
from pandas import DataFrame
import numpy as np


def calculate_gain(row, original_f1_score: float):
    return row['predicted_poly_reg_f1'] - original_f1_score


class ArtificialPollution:
    def __init__(self, metadata, database_url, pre_pollution_setting_id, modifier):
        self.metadata = metadata
        self.DATABASE_URL = database_url
        self.pre_pollution_setting_id = pre_pollution_setting_id
        self.modifier = modifier

    def artificial_pollution(self, history_df: DataFrame, ds_name: str, experiment,
                             feature_wise_pollution_level: dict, iteration: int, skip_pollution=False):
        feature_candidates_for_cleaning = self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols']
        feature_candidates_for_cleaning = self.filter_features_on_type(feature_candidates_for_cleaning, self.metadata, ds_name)

        polluted_dfs = []
        current_f1_score = 0.0
        for feature in feature_candidates_for_cleaning:
            current_pollution_level_train = feature_wise_pollution_level['train'][feature]
            current_pollution_level_test = feature_wise_pollution_level['test'][feature]
            current_pollution_level = {'train': current_pollution_level_train, 'test': current_pollution_level_test}

            if current_pollution_level['train'] == 0 and current_pollution_level['test'] == 0:
                continue

            results_df = pd.DataFrame(columns=['pollution_level', 'real_f1', 'predicted_lin_reg_f1',
                                               'predicted_poly_reg_f1'])

            filtered_history_df, is_empty = self.get_filtered_history_df(history_df, 277712)

            train_df_polluted, test_df_polluted = self.get_current_polluted_training_and_test_df(filtered_history_df, ds_name, feature_wise_pollution_level)
            train_df, test_df = self.get_clean_training_and_test_df(filtered_history_df, ds_name)

            results_df = self.execute_cleaning(current_pollution_level, ds_name, experiment,  feature, results_df,
                                               filtered_history_df, test_df_polluted, train_df_polluted)
            current_f1_score = results_df[results_df['pollution_level'] == feature_wise_pollution_level['train'][feature]]['real_f1'].values[0]

            if not skip_pollution:
                results_df = self.execute_pollution(current_pollution_level, ds_name, experiment, feature,
                                                    results_df, test_df, test_df_polluted, train_df, train_df_polluted)

                regression_model = RegressionModel(results_df, feature_wise_pollution_level, feature)
                results_df = regression_model.fit_regression_models()
                # calculate gain for each pollution level and add it to the dataframe
                results_df['f1_gain_predicted'] = results_df.apply(lambda row: calculate_gain(row, current_f1_score), axis=1)
                self.write_validation_results_to_db(results_df, ds_name, feature, experiment.name, iteration)
            results_df['iteration'] = iteration

            print('Results for feature: ' + feature)
            print(results_df.head(25).to_string())

            polluted_dfs.append({'feature': feature, 'polluted_df': results_df})
        return polluted_dfs, current_f1_score

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
            current_pollution_level = feature_wise_pollution_level['train'][current_feature]
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
            current_pollution_level = feature_wise_pollution_level['test'][current_feature]
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

    def execute_pollution(self, current_pollution_level: dict, ds_name: str, experiment, feature: str, results_df: DataFrame, test_df: DataFrame,
                          test_df_polluted: DataFrame, train_df: DataFrame, train_df_polluted: DataFrame):

        pollution_configs = []

        for random_seed in [87263, 53219, 78604, 2023, 38472, 11, 9834, 4567, 909090, 56789]:
            np.random.seed(random_seed)
            corrupted_train_df = pd.DataFrame(columns=train_df.columns)
            corrupted_test_df = pd.DataFrame(columns=test_df.columns)
            for pollution_level in [0.01, 0.02]:
                pollution_level = round(pollution_level, 2)
                pollution_level_setting = {feature: pollution_level}
                if current_pollution_level['train'] > 0:
                    corrupted_train_df = self.modifier(pollution_level_setting, corrupted_train_df, train_df_polluted,
                                                       self.metadata[ds_name]).pollute2()
                else:
                    corrupted_train_df = train_df.copy()
                if current_pollution_level['test'] > 0:
                    corrupted_test_df = self.modifier(pollution_level_setting, corrupted_test_df, test_df_polluted,
                                                      self.metadata[ds_name]).pollute2()
                else:
                    corrupted_test_df = test_df.copy()

                pollution_configs.append({'pollution_level': current_pollution_level['train']+pollution_level, 'random_seed': random_seed, 'train_df': corrupted_train_df, 'test_df': corrupted_test_df})

        result = Parallel(n_jobs=10)(
            delayed(self.parallel_model_performance_calculation)(pollution_config['random_seed'], pollution_config['pollution_level'], experiment, pollution_config['train_df'], pollution_config['test_df'], ds_name)
            for pollution_config in pollution_configs)
        results_df = pd.concat([results_df, pd.concat(result)], ignore_index=True)

        return results_df

    def execute_cleaning(self, current_pollution_level: dict, ds_name: str, experiment, feature: str, results_df: DataFrame, filtered_history_df: DataFrame,
                         test_df_pre_polluted: DataFrame, train_df_pre_polluted: DataFrame):

        for pollution_level in [{'train': current_pollution_level['train']-0.01, 'test': current_pollution_level['test']-0.01},
                                {'train': current_pollution_level['train'], 'test': current_pollution_level['test']}]:
            #pollution_level = round(pollution_level, 2)
            if pollution_level['train'] < 0:
                pollution_level['train'] = 0.0
            if pollution_level['test'] < 0:
                pollution_level['test'] = 0.0
            pollution_level_train = round(pollution_level['train'], 2)
            pollution_level_test = round(pollution_level['test'], 2)

            df_polluted = filtered_history_df[filtered_history_df['pollution_level'] == pollution_level_train]
            df_polluted = df_polluted.reset_index(drop=True)
            train_df_polluted = df_polluted[df_polluted['train/test'] == 'train']

            # delete index
            train_df_polluted = train_df_polluted.reset_index(drop=True)
            train_df_pre_polluted = train_df_pre_polluted.reset_index(drop=True)
            train_df_pre_polluted[feature] = train_df_polluted[feature]

            train_df_polluted = train_df_pre_polluted.copy()

            train_df_polluted = train_df_polluted[
                self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [
                    self.metadata[ds_name]['target']]]

            df_polluted = filtered_history_df[filtered_history_df['pollution_level'] == pollution_level_test]
            df_polluted = df_polluted.reset_index(drop=True)
            test_df_polluted = df_polluted[df_polluted['train/test'] == 'test']

            test_df_polluted = test_df_polluted.reset_index(drop=True)
            test_df_pre_polluted = test_df_pre_polluted.reset_index(drop=True)
            test_df_pre_polluted[feature] = test_df_polluted[feature]

            test_df_polluted = test_df_pre_polluted.copy()

            test_df_polluted = test_df_polluted[
                self.metadata[ds_name]['categorical_cols'] + self.metadata[ds_name]['numerical_cols'] + [
                    self.metadata[ds_name]['target']]]

            result = Parallel(n_jobs=10)(
                delayed(self.parallel_model_performance_calculation)(random_seed, pollution_level_train, experiment, train_df_polluted, test_df_polluted, ds_name)
                for random_seed in [87263, 53219, 78604, 2023, 38472, 11, 9834, 4567, 909090, 56789])
                #for random_seed in [87263])
            results_df = pd.concat([results_df, pd.concat(result)], ignore_index=True)

        return results_df

    def parallel_model_performance_calculation(self, random_seed: int, pollution_level: float, experiment, train_df_polluted: DataFrame, test_df_polluted: DataFrame, ds_name: str):
        np.random.seed(random_seed)

        exp = experiment(train_df_polluted, test_df_polluted, self.metadata[ds_name], self.modifier.get_classname(), self.pre_pollution_setting_id)
        results = exp.run('', 'scenario', explain=False)

        new_results_row = pd.Series({'pollution_level': pollution_level,
                                     'real_f1': results[exp.name]['scoring']['macro avg']['f1-score'],
                                     'predicted_lin_reg_f1': 0.0,
                                     'predicted_poly_reg_f1': 0.0})

        return new_results_row.to_frame().T

    def write_validation_results_to_db(self, results_df: DataFrame, ds_name: str, feature: str, exp_name: str, iteration: int, if_exists='replace'):
        if self.pre_pollution_setting_id is None:
            return None

        modifier_str = self.modifier.get_classname()
        results_df_copy = results_df.copy()
        results_df_copy['iteration'] = iteration
        results_df_copy['cleaning_level'] = results_df_copy['pollution_level']
        results_df_copy = results_df_copy.drop(columns=['pollution_level'], axis=1)

        complete_results = load_data_from_db(
            f'validation_results_{ds_name}_{exp_name}_{modifier_str}_{feature}_{self.pre_pollution_setting_id}', self.DATABASE_URL)

        if complete_results is not None and 'iteration' in complete_results.columns:
            complete_results = complete_results[complete_results['iteration'] != iteration]

        complete_results = pd.concat([complete_results, results_df_copy])
        DATABASE_ENGINE = create_engine(self.DATABASE_URL, echo=False, connect_args={'timeout': 2000})
        try:
            complete_results.to_sql(
                name=f'validation_results_{ds_name}_{exp_name}_{modifier_str}_{feature}_{self.pre_pollution_setting_id}',
                con=DATABASE_ENGINE, if_exists=if_exists, index=False)
        except Exception as e:
            print(e)

    def clean_from_cleaning_buffer(self, cleaning_buffer: set, history_df: DataFrame, ds_name: str, experiment,
                             feature_wise_pollution_level: dict):
        # decrease pollution level by 0.01, according to the cleaning buffer, stop at 0.0
        pollution_levels = feature_wise_pollution_level.copy()
        for feature in cleaning_buffer:
            current_pollution_level = pollution_levels.copy()
            if current_pollution_level['train'][feature] == 0.0 and current_pollution_level['test'][feature] == 0.0:
                continue
            pollution_levels['train'][feature] = round(current_pollution_level['train'][feature] - 0.01, 2)
            pollution_levels['test'][feature] = round(current_pollution_level['test'][feature] - 0.01, 2)

        filtered_history_df, is_empty = self.get_filtered_history_df(history_df, 277712)
        train_df_polluted, test_df_polluted = self.get_current_polluted_training_and_test_df(filtered_history_df, ds_name, pollution_levels)

        np.random.seed(87263)
        exp = experiment(train_df_polluted, test_df_polluted, self.metadata[ds_name], self.modifier.get_classname(), self.pre_pollution_setting_id)
        results = exp.run('', 'scenario', explain=False)
        return results[exp.name]['scoring']['macro avg']['f1-score']
