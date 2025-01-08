from pandas import DataFrame, Series
from jenga.corruptions.numerical import GaussianNoise, Scaling
from jenga.corruptions.generic import MissingValues, CategoricalShift
import pandas as pd
from jenga.corruptions.generic import TabularCorruption
import numpy as np
import warnings


def make_columns_numeric(df, pollution_level_setting):
    for feature in pollution_level_setting.keys():
        df[feature] = pd.to_numeric(df[feature])


class DatasetModifier:

    def __init__(self, jenga_polluter, features_to_pollute: dict, original_data: DataFrame, data: DataFrame, **kwargs):
        self.jenga_polluter = jenga_polluter
        self.features_to_pollute = features_to_pollute
        self.original_data = original_data.copy()
        self.data = data.copy()
        self.jenga_config = kwargs

    def pollute(self, keep_track=False):
        dirty_indexes_test = {}
        for feature, pollution_level in self.features_to_pollute.items():
            current_config = {}
            for param, value in self.jenga_config.items():
                if feature in value:
                    current_config[param] = value[feature]

            pollutable_data = self.get_pollutable_data(feature)
            real_pollution_level = self.calculate_real_pollution_level(pollutable_data, pollution_level)
            temp_df = pollutable_data.copy()
            pollutable_data[feature] = self.jenga_polluter(column=feature, fraction=real_pollution_level, **current_config).transform(pollutable_data)[feature]
            self.data.loc[pollutable_data.index.values, feature] = pollutable_data[feature][pollutable_data.index.values]

            mismatches: Series = temp_df[feature] != pollutable_data[feature]
            dirty_indexes_test[feature] = mismatches[mismatches].index.tolist()
            print('number of polluted rows: ', len(dirty_indexes_test))
        print(' ')
        if keep_track:
            return self.data, dirty_indexes_test
        return self.data

    def pollute2(self):
        for feature, pollution_level in self.features_to_pollute.items():
            current_config = {}
            for param, value in self.jenga_config.items():
                if feature in value:
                    current_config[param] = value[feature]
            pollutable_data = self.data
            real_pollution_level = pollution_level
            pollutable_data[feature] = self.jenga_polluter(column=feature, fraction=real_pollution_level, **current_config).transform(pollutable_data)[feature]
            self.data.loc[pollutable_data.index.values, feature] = pollutable_data[feature][pollutable_data.index.values]

        return self.data

    def pollute3(self, keep_track=False):
        dirty_indexes_test = {}
        for feature, pollution_level in self.features_to_pollute.items():
            current_config = {}
            for param, value in self.jenga_config.items():
                if feature in value:
                    current_config[param] = value[feature]

            error_map_train_df = self.data != self.original_data
            still_clean_data_indexes = error_map_train_df[error_map_train_df[feature] == False].index.tolist()
            pollutable_data = self.data.iloc[still_clean_data_indexes]
            real_pollution_level = self.calculate_real_pollution_level(pollutable_data, pollution_level)
            temp_df = pollutable_data.copy()
            pollutable_data[feature] = self.jenga_polluter(column=feature, fraction=real_pollution_level, **current_config).transform(pollutable_data)[feature]
            self.data.loc[pollutable_data.index.values, feature] = pollutable_data[feature][pollutable_data.index.values]

            mismatches: Series = temp_df[feature] != pollutable_data[feature]
            dirty_indexes_test[feature] = mismatches[mismatches].index.tolist()
            print('feature', feature, 'number of polluted rows: ', len(dirty_indexes_test[feature]), 'error type: ', self.jenga_polluter.__name__)
        print(' ')
        if keep_track:
            return self.data, dirty_indexes_test
        return self.data

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def calculate_real_pollution_level(self, pollutable_data, pollution_level):
        original_length = len(self.original_data)
        already_polluted_rows = original_length - len(pollutable_data)
        pollutable_rows = len(pollutable_data)
        number_of_polluted_rows_after_pollution = original_length * pollution_level
        rows_to_pollute = int(number_of_polluted_rows_after_pollution - already_polluted_rows)
        real_pollution_level = rows_to_pollute / pollutable_rows
        return real_pollution_level

    def get_pollutable_data(self, feature):
        self.original_data = self.original_data.reset_index(drop=True)
        self.data = self.data.reset_index(drop=True)
        ne: Series = Series(self.original_data[feature] != self.data[feature])
        pollutable_data = self.data.iloc[ne[ne == False].index.values]
        return pollutable_data


def delete_non_numerical_features_from_pollution_setting(metadata, pollution_level_setting):
    for feature in list(pollution_level_setting.keys()):
        if feature not in metadata['numerical_cols']:
            del pollution_level_setting[feature]


def delete_non_categorical_features_from_pollution_setting(metadata, pollution_level_setting):
    for feature in list(pollution_level_setting.keys()):
        if feature not in metadata['categorical_cols']:
            del pollution_level_setting[feature]


class ModifiedGaussianNoise(TabularCorruption):

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def transform(self, data):
        df = data.copy(deep=True)
        stddev = np.std(df[self.column])
        scale = np.random.uniform(1, 5)

        if self.fraction > 0:
            rows = self.sample_rows(data)
            noise = np.random.normal(0, scale * stddev, size=len(rows))
            df.loc[rows, self.column] += noise

        return df


class GaussianNoiseModifier(DatasetModifier):
    restricted_to = 'numerical_col'

    def __init__(self, pollution_level_setting, original_df, polluted_df, metadata):
        sampling = 'MCAR'

        jenga_config = {'sampling': {}}
        for feature in pollution_level_setting.keys():
            jenga_config['sampling'][feature] = sampling

        delete_non_numerical_features_from_pollution_setting(metadata, pollution_level_setting)
        make_columns_numeric(original_df, pollution_level_setting)
        make_columns_numeric(polluted_df, pollution_level_setting)

        super().__init__(ModifiedGaussianNoise, pollution_level_setting, original_df, polluted_df, **jenga_config)


class MissingValuesModifier(DatasetModifier):
    restricted_to = ''

    def __init__(self, pollution_level_setting, original_df, polluted_df, metadata):
        sampling = 'MCAR'

        jenga_config = {'missingness': {}, 'na_value': {}}
        for feature in pollution_level_setting.keys():
            jenga_config['missingness'][feature] = sampling

            if feature in metadata['categorical_cols']:
                feature_type = 'categorical'
            else:
                feature_type = 'numerical'

            if type(metadata['placeholders'][feature_type]) != dict:
                jenga_config['na_value'][feature] = metadata['placeholders'][feature_type]
            else:
                jenga_config['na_value'][feature] = metadata['placeholders'][feature_type][feature]

        super().__init__(MissingValues, pollution_level_setting, original_df, polluted_df, **jenga_config)


class ScalingModifier(DatasetModifier):
    restricted_to = 'numerical_col'

    def __init__(self, pollution_level_setting, original_df, polluted_df, metadata):
        sampling = 'MCAR'

        jenga_config = {'sampling': {}}
        for feature in pollution_level_setting.keys():
            jenga_config['sampling'][feature] = sampling

        delete_non_numerical_features_from_pollution_setting(metadata, pollution_level_setting)
        make_columns_numeric(original_df, pollution_level_setting)
        make_columns_numeric(polluted_df, pollution_level_setting)

        super().__init__(Scaling, pollution_level_setting, original_df, polluted_df, **jenga_config)


class ModifiedCategoricalShift(TabularCorruption):
    def transform2(self, data):
        df = data.copy(deep=True)
        rows = self.sample_rows(df)

        histogram = df[self.column].value_counts()
        random_other_val = np.random.permutation(histogram.index)
        df.loc[rows, self.column] = df.loc[rows, self.column].replace(histogram.index, random_other_val)
        return df

    def transform(self, data):
        if self.fraction == 0:
            return data
        df = data.copy(deep=True)
        rows = self.sample_rows(df)
        # check if there are enough unique values to shuffle
        if len(df.loc[rows, self.column].unique()) < 2:
            for i in range(10):
                rows = self.sample_rows(df)
                if len(df.loc[rows, self.column].unique()) >= 2:
                    break


        shuffled_arr = np.array(df.loc[rows, self.column].values)  # Create a copy to shuffle
        np.random.shuffle(shuffled_arr)  # Shuffle the copy

        result = np.empty_like(df.loc[rows, self.column].values)

        for i, (original_val, shuffled_val) in enumerate(zip(df.loc[rows, self.column].values, shuffled_arr)):
            if original_val == shuffled_val:
                # Find the next available value from shuffled_arr
                available_values = shuffled_arr[shuffled_arr != original_val]
                if len(available_values) == 0:
                    warnings.warn("Not enough unique values to shuffle. Skipping this column.")
                    available_values = [original_val]
                next_val = available_values[0]
                result[i] = next_val
            else:
                result[i] = shuffled_val

        df.loc[rows, self.column] = result
        return df


class CategoricalShiftModifier(DatasetModifier):
    restricted_to = 'categorical_col'

    def __init__(self, pollution_level_setting, original_df, polluted_df, metadata):
        sampling = 'MCAR'

        jenga_config = {'sampling': {}}
        for feature in pollution_level_setting.keys():
            jenga_config['sampling'][feature] = sampling

        delete_non_categorical_features_from_pollution_setting(metadata, pollution_level_setting)

        super().__init__(ModifiedCategoricalShift, pollution_level_setting, original_df, polluted_df, **jenga_config)
