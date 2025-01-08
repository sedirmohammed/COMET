from abc import ABC, abstractmethod
import numpy as np
from pandas import Series
from classification.utils.artifical_pollution_new import *


class CleaningStrategy(ABC):

    @abstractmethod
    def select_cleaning_setting(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_cleaning_buffer(self):
        """Return the cleaning buffer. If not applicable, return None."""
        return None


def _filter_polluted_df(feature_entry, feature_wise_pollution_level, budget):
    """Filter and preprocess polluted dataframe based on criteria"""
    current_df = feature_entry['polluted_df'].copy()
    # round pollution level to 2 decimal places
    current_df['pollution_level'] = current_df['pollution_level'].apply(lambda x: round(x, 2))
    current_df = current_df[current_df['pollution_level'] == current_df['pollution_level'].min()]

    # Convert types and filter entries based on budget
    current_df['predicted_poly_reg_f1'] = current_df['predicted_poly_reg_f1'].astype(float)
    current_df['used_budget'] = current_df['used_budget'].astype(float)
    current_df = current_df[current_df['used_budget'] <= budget]

    # Sort and select top entry
    current_df = current_df.sort_values(by=['predicted_poly_reg_f1'], ascending=False).head(1)

    return current_df


def get_current_f1_score(feature_entry, feature_wise_pollution_level) -> float:
    """Get current f1 score based on feature and pollution level"""
    current_df = feature_entry['polluted_df'].copy()
    current_df = current_df[current_df['pollution_level'] == feature_wise_pollution_level['train'][feature_entry['feature']]]
    return current_df['real_f1'].values[0]


def _update_cleaning_setting(cleaning_setting, current_df, f1_gain_predicted, feature):
    """Update cleaning settings if the current dataframe has a better prediction score"""
    lambda_ = 1.0
    prediction_score = current_df['predicted_poly_reg_f1'].values[0] - lambda_ * (
            current_df['upper_confidence_border'].values[0] - current_df['lower_confidence_border'].values[0]
    )
    if cleaning_setting['prediction_score'] < prediction_score:
        cleaning_setting['feature'] = feature
        cleaning_setting['pollution_level'] = current_df['pollution_level'].values[0]
        cleaning_setting['predicted_poly_reg_f1'] = current_df['predicted_poly_reg_f1'].values[0]
        cleaning_setting['real_f1'] = current_df['real_f1'].values[0]
        cleaning_setting['used_budget'] = round(current_df['used_budget'].values[0], 0)
        cleaning_setting['f1_gain_predicted'] = f1_gain_predicted
        cleaning_setting['prediction_score'] = prediction_score


def _select_based_on_importance(features_importance, feature_wise_pollution_level, polluted_dfs, budget):
    """Select the next feature to clean based on its importance."""
    # Find the next feature to clean based on its importance and if it hasn't been cleaned yet
    for feature in features_importance.to_dict().keys():
        if feature_wise_pollution_level['train'][feature] > 0 or feature_wise_pollution_level['test'][feature] > 0:
            # get list entry where feature is the key
            feature_entry = next((item for item in polluted_dfs if item['feature'] == feature), None)
            if feature_entry is None:
                continue
            print('feature', feature)
            print(feature_entry['polluted_df'])
            current_df = _filter_polluted_df(feature_entry, feature_wise_pollution_level, budget)

            return {
                'feature': feature,
                'pollution_level': current_df['pollution_level'].values[0],
                'predicted_poly_reg_f1': 0.5,
                'real_f1': current_df['real_f1'].values[0],
                'used_budget': round(current_df['used_budget'].values[0], 0),
                'f1_gain_predicted': -1,
                'prediction_score': -1
            }


def update_prediction_history(feature, predicted_value, actual_value, prediction_accuracy_history):
    #percentage_error = (predicted_value - actual_value) / actual_value if actual_value != 0 else 0
    if actual_value > predicted_value:
        absolute_error = actual_value - predicted_value
        percentage_error = absolute_error / actual_value
    else:
        absolute_error = predicted_value - actual_value
        percentage_error = absolute_error / predicted_value
    prediction_accuracy_history[feature].append(absolute_error)


def get_last_percentage_error(feature, prediction_accuracy_history):
    if prediction_accuracy_history[feature]:
        return np.mean(prediction_accuracy_history[feature])
    else:
        return 0


def adjust_prediction_with_last_error(feature, current_prediction, prediction_accuracy_history):
    last_percentage_error = get_last_percentage_error(feature, prediction_accuracy_history)
    #return current_prediction * (1 - last_percentage_error), last_percentage_error
    return current_prediction + last_percentage_error, last_percentage_error


class CleaningSetting5(CleaningStrategy):

    def __init__(self):
        self.cleaning_buffer = set()

    def get_cleaning_buffer(self):
        return self.cleaning_buffer

    def select_cleaning_setting(self, polluted_dfs, feature_wise_pollution_level, budget, **kwargs):
        prediction_accuracy_history = kwargs.get('prediction_accuracy_history', {})
        features_importance = kwargs.get('features_importance', {})

        # Initialize default cleaning setting
        cleaning_setting = {
            'feature': None,
            'pollution_level': -1,
            'predicted_poly_reg_f1': -1,
            'real_f1': -1,
            'used_budget': -1,
            'f1_gain_predicted': -1,
            'prediction_score': -1
        }

        adjusted_predictions = {}
        positive_predictions = []

        for feature_entry in polluted_dfs:
            current_df = _filter_polluted_df(feature_entry, feature_wise_pollution_level, budget)

            if current_df.empty:
                print('current_polluted_df is empty')
                continue

            current_prediction = current_df['predicted_poly_reg_f1'].values[0]

            adjusted_pred, error = adjust_prediction_with_last_error(
                feature_entry['feature'], current_prediction, prediction_accuracy_history
            )
            current_df['f1_gain_predicted'] = current_df['f1_gain_predicted'].values[0] + error
            current_df['predicted_poly_reg_f1'] = adjusted_pred
            adjusted_predictions[feature_entry['feature']] = adjusted_pred

            cleaning_setting['feature'] = feature_entry['feature']
            cleaning_setting['pollution_level'] = current_df['pollution_level'].values[0]
            cleaning_setting['predicted_poly_reg_f1'] = current_df['predicted_poly_reg_f1'].values[0]
            cleaning_setting['real_f1'] = current_df['real_f1'].values[0]
            cleaning_setting['used_budget'] = round(current_df['used_budget'].values[0], 0)
            cleaning_setting['f1_gain_predicted'] = current_df['f1_gain_predicted'].values[0]
            cleaning_setting['prediction_score'] = current_df['predicted_poly_reg_f1'].values[0] - 1.0 * (
                    current_df['upper_confidence_border'].values[0] - current_df['lower_confidence_border'].values[0])

            if current_df['f1_gain_predicted'].values[0] > 0:
                positive_predictions.append(cleaning_setting.copy())

        positive_predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
        # Start by checking the best setting
        current_f1_score = get_current_f1_score(polluted_dfs[0], feature_wise_pollution_level)
        used_budget = 0
        max_real_f1 = 0.0
        index_of_best_setting = 0
        found_setting = False
        if len(positive_predictions) > 0:
            for setting in positive_predictions:
                if not setting['feature'] in self.cleaning_buffer:
                    used_budget += 1
                    self.cleaning_buffer.add(setting['feature'])
                setting['used_budget'] = used_budget
                if setting['real_f1'] > max_real_f1:
                    max_real_f1 = setting['real_f1']
                    index_of_best_setting = positive_predictions.index(setting)
                if setting['real_f1'] >= current_f1_score or used_budget == budget:
                    cleaning_setting = positive_predictions[index_of_best_setting]
                    found_setting = True
                    self.cleaning_buffer.remove(setting['feature'])
                    break

        if len(positive_predictions) == 0 or not found_setting:
            cleaning_setting = _select_based_on_importance(features_importance, feature_wise_pollution_level, polluted_dfs, budget)
            if cleaning_setting['feature'] in self.cleaning_buffer:
                self.cleaning_buffer.remove(cleaning_setting['feature'])
                cleaning_setting['used_budget'] = used_budget
            else:
                cleaning_setting['used_budget'] = cleaning_setting['used_budget'] + used_budget

        update_prediction_history(
            cleaning_setting['feature'],
            adjusted_predictions[cleaning_setting['feature']],
            cleaning_setting['real_f1'],
            prediction_accuracy_history
        )
        return cleaning_setting


class CleaningSetting6(CleaningStrategy):

    def __init__(self):
        self.cleaning_buffer = set()

    def get_cleaning_buffer(self):
        return self.cleaning_buffer

    def select_cleaning_setting(self, ml_algorithm, ap: ArtificialPollution2, polluted_dfs, feature_wise_pollution_level, budget, **kwargs):
        prediction_accuracy_history = kwargs.get('prediction_accuracy_history', {})
        features_importance = kwargs.get('features_importance', {})

        # Initialize default cleaning setting
        cleaning_setting = {
            'feature': None,
            'pollution_level': -1,
            'predicted_poly_reg_f1': -1,
            'real_f1': -1,
            'used_budget': -1,
            'f1_gain_predicted': -1,
            'prediction_score': -1
        }

        adjusted_predictions = {}
        positive_predictions = []

        for feature_entry in polluted_dfs:
            current_df = _filter_polluted_df(feature_entry, feature_wise_pollution_level, budget)

            feature = feature_entry['feature']
            current_train_df = feature_entry['pollution_configs'][0]['train_df']
            current_test_df = feature_entry['pollution_configs'][0]['test_df']
            train_df = ap.train_df
            test_df = ap.test_df

            sampling_indexes_train = []
            sampling_indexes_test = []
            for pollution_config in feature_entry['pollution_configs']:
                sampling_indexes_train.extend(pollution_config['sampling_indexes_train'])
                sampling_indexes_test.extend(pollution_config['sampling_indexes_test'])
            # remove duplicates
            sampling_indexes_train = list(dict.fromkeys(sampling_indexes_train))
            sampling_indexes_test = list(dict.fromkeys(sampling_indexes_test))

            # get index where current_train_df and train_df are different for feature feature_entry['feature']

            current_train_df = self.clean_df_by_one_cleaning_step(current_train_df, feature, sampling_indexes_train, train_df)
            current_test_df = self.clean_df_by_one_cleaning_step(current_test_df, feature, sampling_indexes_test, test_df)

            np.random.seed(42)
            exp = ml_algorithm(current_train_df, current_test_df, ap.metadata[ap.ds_name], ap.modifier, ap.pre_pollution_setting_id)
            results = exp.run('', 'scenario', explain=False)
            current_df['real_f1'].values[0] = results[exp.name]['scoring']['macro avg']['f1-score']
            current_df['used_budget'].values[0] = 1

            if current_df.empty:
                print('current_polluted_df is empty')
                continue

            current_prediction = current_df['predicted_poly_reg_f1'].values[0]

            adjusted_pred, error = adjust_prediction_with_last_error(
                feature_entry['feature'], current_prediction, prediction_accuracy_history
            )
            current_df['f1_gain_predicted'] = current_df['f1_gain_predicted'].values[0] + error
            current_df['predicted_poly_reg_f1'] = adjusted_pred
            adjusted_predictions[feature_entry['feature']] = adjusted_pred

            cleaning_setting['feature'] = feature_entry['feature']
            cleaning_setting['pollution_level'] = current_df['pollution_level'].values[0]
            cleaning_setting['predicted_poly_reg_f1'] = current_df['predicted_poly_reg_f1'].values[0]
            cleaning_setting['real_f1'] = current_df['real_f1'].values[0]
            cleaning_setting['used_budget'] = round(current_df['used_budget'].values[0], 0)
            cleaning_setting['f1_gain_predicted'] = current_df['f1_gain_predicted'].values[0]
            cleaning_setting['prediction_score'] = current_df['predicted_poly_reg_f1'].values[0] - 1.0 * (
                    current_df['upper_confidence_border'].values[0] - current_df['lower_confidence_border'].values[0])

            if current_df['f1_gain_predicted'].values[0] > 0:
                positive_predictions.append(cleaning_setting.copy())

        positive_predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
        # Start by checking the best setting
        current_f1_score = get_current_f1_score(polluted_dfs[0], feature_wise_pollution_level)
        used_budget = 0
        max_real_f1 = 0.0
        index_of_best_setting = 0
        found_setting = False
        if len(positive_predictions) > 0:
            for setting in positive_predictions:
                if not setting['feature'] in self.cleaning_buffer:
                    used_budget += 1
                    self.cleaning_buffer.add(setting['feature'])
                setting['used_budget'] = used_budget
                if setting['real_f1'] > max_real_f1:
                    max_real_f1 = setting['real_f1']
                    index_of_best_setting = positive_predictions.index(setting)
                if setting['real_f1'] >= current_f1_score or used_budget == budget:
                    cleaning_setting = positive_predictions[index_of_best_setting]
                    found_setting = True
                    self.cleaning_buffer.remove(setting['feature'])
                    break

        if len(positive_predictions) == 0 or not found_setting:
            cleaning_setting = _select_based_on_importance(features_importance, feature_wise_pollution_level, polluted_dfs, budget)
            if cleaning_setting['feature'] in self.cleaning_buffer:
                self.cleaning_buffer.remove(cleaning_setting['feature'])
                cleaning_setting['used_budget'] = used_budget
            else:
                cleaning_setting['used_budget'] = cleaning_setting['used_budget'] + used_budget

        update_prediction_history(
            cleaning_setting['feature'],
            adjusted_predictions[cleaning_setting['feature']],
            cleaning_setting['real_f1'],
            prediction_accuracy_history
        )
        return cleaning_setting

    def clean_df_by_one_cleaning_step(self, current_polluted_df, feature, sampling_indexes, clean_df):
        ne: Series = Series(current_polluted_df[feature] != clean_df[feature])
        dirty_indexes_train = ne[ne == True].index.values.tolist()
        # match dirty indexes with sampling indexes
        cleaning_indexes = [x for x in dirty_indexes_train if x in sampling_indexes]
        number_of_rows_to_clean_train = int(len(clean_df) * 0.01)
        if len(cleaning_indexes) > number_of_rows_to_clean_train:
            cleaning_indexes = cleaning_indexes[:number_of_rows_to_clean_train]
        else:
            remaining_rows_to_clean = number_of_rows_to_clean_train - len(cleaning_indexes)
            np.random.seed(42)
            random_cleaning_indexes = np.random.choice(
                [x for x in dirty_indexes_train if x not in cleaning_indexes], remaining_rows_to_clean,
                replace=False)
            cleaning_indexes.extend(random_cleaning_indexes)
        # take from train_df from feature feature_entry['feature'] the cells at indexes cleaning_indexes_train and store them in current_train_df
        current_polluted_df.loc[cleaning_indexes, feature] = clean_df.loc[cleaning_indexes, feature]
        return current_polluted_df
