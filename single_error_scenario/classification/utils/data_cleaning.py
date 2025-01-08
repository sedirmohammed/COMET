from abc import ABC, abstractmethod
import random
from pandas import Series, DataFrame
from typing import Dict, List, TypedDict


class CleaningInterface(ABC):
    @abstractmethod
    def clean(self, feature, **kwargs):
        pass


class CleaningConfigType(TypedDict):
    data: DataFrame
    cleaning_candidates_train: List[int]
    cleaning_candidates_test: List[int]


def get_pollutable_data(polluted_df, mismatches):
    pollutable_data = polluted_df.iloc[mismatches[mismatches == True].index.values]
    return pollutable_data


def calculate_real_pollution_level(clean_df, pollutable_data, cleaning_step):
    original_length = len(clean_df)
    number_of_polluted_rows = len(pollutable_data)
    number_of_cleaned_rows_fraction = original_length * cleaning_step
    number_polluted_rows_after_next_cleaning = int(number_of_polluted_rows - number_of_cleaned_rows_fraction)
    return 0.0, max(0, number_of_polluted_rows-number_polluted_rows_after_next_cleaning)  # Ensure we don't end up with a negative number


def calculate_real_pollution_level2(clean_df, pollutable_data, cleaning_step):
    original_length = len(clean_df)  # The size of the full, clean dataset
    number_of_polluted_rows = len(pollutable_data)  # Current count of polluted rows

    # Calculate the number of rows to clean in this step as a fraction of the original dataset size
    number_of_cleaned_rows_fraction = original_length * cleaning_step

    # Ensure we don't plan to clean more rows than are currently polluted
    number_of_cleaned_rows_fraction = min(number_of_cleaned_rows_fraction, number_of_polluted_rows)

    # Calculate how many polluted rows would remain after this cleaning step
    # and the actual number of rows we plan to clean in this step
    number_of_polluted_rows_after_next_cleaning = max(0, number_of_polluted_rows - int(number_of_cleaned_rows_fraction))
    rows_to_clean_in_next_step = number_of_polluted_rows - number_of_polluted_rows_after_next_cleaning

    return 0.0, rows_to_clean_in_next_step


class SimulationCleaning(CleaningInterface):
    def clean(self, feature: str, **kwargs):
        polluted_train_df: DataFrame = kwargs.get('train_df_polluted').copy()
        polluted_test_df: DataFrame = kwargs.get('test_df_polluted').copy()
        train_df: DataFrame = kwargs.get('train_df')
        test_df: DataFrame = kwargs.get('test_df')
        cleaning_config: CleaningConfigType = kwargs.get('cleaning_config')
        cleaning_candidates_train: List = cleaning_config['cleaning_candidates_train']
        cleaning_candidates_test: List = cleaning_config['cleaning_candidates_test']
        cleaning_step_size = kwargs.get('cleaning_step_size')

        random.seed(42)
        cleaning_results = {'train': {}, 'test': {}}
        if cleaning_candidates_train is not None:
            mismatches: Series = polluted_train_df[feature] != train_df[feature]
            dirty_indexes_train = mismatches[mismatches].index.tolist()

            pollutable_data = get_pollutable_data(polluted_train_df, mismatches)
            real_pollution_level, number_rows_to_pollute = calculate_real_pollution_level(train_df, pollutable_data, cleaning_step_size)
            if len(dirty_indexes_train) > 0:
                cleaning_candidates_train = [index for index in cleaning_candidates_train if index in dirty_indexes_train]
                cleaning_candidates_train: set = set(cleaning_candidates_train)
                for index in cleaning_candidates_train:
                    dirty_indexes_train.remove(index)
                while len(cleaning_candidates_train) < number_rows_to_pollute:
                    if len(dirty_indexes_train) == 0:
                        break
                    cleaning_candidates_train.add(dirty_indexes_train.pop())
                cleaning_candidates_train: List = list(cleaning_candidates_train)
                if len(dirty_indexes_train) < number_rows_to_pollute:
                    #cleaning_candidates_train.extend(dirty_indexes_train)
                    print()
                else:
                    if len(cleaning_candidates_train) > number_rows_to_pollute:
                        cleaning_candidates_train = random.sample(cleaning_candidates_train, number_rows_to_pollute)
                polluted_train_df.loc[cleaning_candidates_train, feature] = train_df.loc[cleaning_candidates_train, feature]
                cleaning_results['train']['data'] = polluted_train_df.copy()
                cleaning_results['train']['indexes'] = cleaning_candidates_train
            else:
                cleaning_results['train']['data'] = polluted_train_df.copy()
                cleaning_results['train']['indexes'] = None


        if cleaning_candidates_test is not None:
            mismatches: Series = polluted_test_df[feature] != test_df[feature]
            dirty_indexes_test = mismatches[mismatches].index.tolist()

            pollutable_data = get_pollutable_data(polluted_test_df, mismatches)
            real_pollution_level, number_rows_to_pollute = calculate_real_pollution_level(test_df, pollutable_data, cleaning_step_size)
            if len(dirty_indexes_test) > 0:
                cleaning_candidates_test = [index for index in cleaning_candidates_test if index in dirty_indexes_test]
                cleaning_candidates_test: set = set(cleaning_candidates_test)
                for index in cleaning_candidates_test:
                    dirty_indexes_test.remove(index)
                while len(cleaning_candidates_test) < number_rows_to_pollute:
                    if len(dirty_indexes_test) == 0:
                        break
                    cleaning_candidates_test.add(dirty_indexes_test.pop())
                cleaning_candidates_test = list(cleaning_candidates_test)
                if len(dirty_indexes_test) < number_rows_to_pollute:
                    #cleaning_candidates_test.extend(dirty_indexes_test)
                    print()
                else:
                    if len(cleaning_candidates_test) > number_rows_to_pollute:
                        cleaning_candidates_test = random.sample(cleaning_candidates_test, number_rows_to_pollute)
                polluted_test_df.loc[cleaning_candidates_test, feature] = test_df.loc[cleaning_candidates_test, feature]
                cleaning_results['test']['data'] = polluted_test_df.copy()
                cleaning_results['test']['indexes'] = cleaning_candidates_test
            else:
                cleaning_results['test']['data'] = polluted_test_df.copy()
                cleaning_results['test']['indexes'] = None
        return cleaning_results


class Cleaner:
    def __init__(self, cleaning_instance: CleaningInterface):
        self.cleaning_instance = cleaning_instance

    def set_cleaning_instance(self, cleaning_instance: CleaningInterface):
        self.cleaning_instance = cleaning_instance

    def clean(self, feature, **kwargs):
        return self.cleaning_instance.clean(feature, **kwargs)
