import time

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from classification.utils.DatasetModifier import *
from classification.utils.artifical_pollution_new import *
from json import load as load_json

from classification.utils.util import load_pre_pollution_df, get_pre_pollution_settings
import argparse
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sqlalchemy import create_engine

from config.definitions import ROOT_DIR
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")

class ClassificationExperiment:

    def __init__(self, name, metadata, model):
        self.name = name
        self.metadata = metadata
        self.model = model

    def fit(self, encoder, ds_name, clean_partial_train_df):
        np.random.seed(42)
        md = self.metadata
        X_train = clean_partial_train_df.drop(md[ds_name]['target'], axis=1)
        y_train = clean_partial_train_df[md[ds_name]['target']]
        if encoder is not None:
            continuous_features_df = X_train.drop(md[ds_name]['categorical_cols'], axis=1)
            transformed_X_train = encoder.transform(X_train[md[ds_name]['categorical_cols']])
            X_train_prepared = np.hstack([transformed_X_train.toarray(), continuous_features_df])
        else:
            X_train_prepared = X_train
        self.model.fit(X_train_prepared, y_train)

    def partial_fit(self, encoder, ds_name, clean_partial_train_df):
        np.random.seed(42)
        md = self.metadata
        X_train = clean_partial_train_df.drop(md[ds_name]['target'], axis=1)
        y_train = clean_partial_train_df[md[ds_name]['target']]
        if encoder is not None:
            continuous_features_df = X_train.drop(md[ds_name]['categorical_cols'], axis=1)
            transformed_X_train = encoder.transform(X_train[md[ds_name]['categorical_cols']])
            X_train_prepared = np.hstack([transformed_X_train.toarray(), continuous_features_df])
        else:
            X_train_prepared = X_train
        self.model.partial_fit(X_train_prepared, y_train)

    def predict(self, encoder, ds_name, clean_partial_test_df):
        np.random.seed(42)
        md = self.metadata
        X_test = clean_partial_test_df.drop(md[ds_name]['target'], axis=1)
        y_test = clean_partial_test_df[md[ds_name]['target']]
        if encoder is not None:
            continuous_features_df = X_test.drop(md[ds_name]['categorical_cols'], axis=1)
            transformed_X_train = encoder.transform(X_test[md[ds_name]['categorical_cols']])
            X_test_prepared = np.hstack([transformed_X_train.toarray(), continuous_features_df])
        else:
            X_test_prepared = X_test
        y_pred = self.model.predict(X_test_prepared)
        results = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        return results['macro avg']['f1-score']


class SGDClassifierExperimentSVM(ClassificationExperiment):

    def __init__(self, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        try:
            hyper_params = metadata[self.__class__.__name__][modifier_name]['pre_pollution_setting_id'][str(pre_pollution_setting_id)]['best_params'].copy()
            for key, value in hyper_params.items():
                if value == 'True':
                    hyper_params[key] = True
                elif value == 'False':
                    hyper_params[key] = False
                if value == 'None':
                    hyper_params[key] = None
        except KeyError:
            hyper_params = {'random_state': 42, 'loss': 'hinge'}

        model = SGDClassifier(**hyper_params)
        super().__init__(self.__class__.__name__, metadata, model)

    def calculate_importance_score(self, encoder, row, ds_name, delta_x, delta_y, dirty_indexes_per_feature, dirty_record_index):
        md = self.metadata
        x = row.drop(md[ds_name]['target'])
        y = row[md[ds_name]['target']]
        # Convert x to dataframe
        x = pd.DataFrame(x).transpose()
        continuous_features_df = x.drop(md[ds_name]['categorical_cols'], axis=1)
        if encoder is not None:
            transformed_X_train = encoder.transform(x[md[ds_name]['categorical_cols']])
            x = np.hstack([transformed_X_train.toarray(), continuous_features_df.values])
        else:
            x = continuous_features_df.values
        if ds_name == 'cmc.data':
            theta = self.model.coef_
            theta = np.mean(theta, axis=0)
        else:
            theta = self.model.coef_.flatten()
        intercept = self.model.intercept_[0]
        decision_function = np.dot(theta, x.T) + intercept  # Adjusted to use x.T for correct alignment
        if y * decision_function <= 1:
            gradient = -y * x.flatten()
        else:
            gradient = np.zeros_like(x.flatten())  # Similarly, flatten the zeros array
        gradient = gradient.reshape((-1, 1))



        n_features = x.shape[1]  # Number of features
        Mx = np.zeros((n_features, n_features))  # Initialize Mx as a zero matrix

        # Calculate the decision function for the record
        decision_function = np.dot(theta, x.T) + intercept

        # Conditionally update the diagonal elements of Mx
        for i in range(n_features):
            if y * decision_function <= 1:
                Mx[i, i] = -y

        My = x.reshape(n_features, 1)


        delta_xr = np.zeros((n_features,1))
        for feature in dirty_indexes_per_feature:
            if feature == md[ds_name]['target']:
                continue
            if dirty_record_index in dirty_indexes_per_feature[feature]:
                feature_index = row.columns.get_loc(feature)
                delta_xr[feature_index, 0] = delta_x[feature_index]
            else:
                feature_index = row.columns.get_loc(feature)
                delta_xr[feature_index, 0] = 0

        delta_yr = delta_y

        p_r = np.linalg.norm(gradient + np.dot(Mx, delta_xr))

        return p_r


class SGDClassifierExperimentLinearRegression(ClassificationExperiment):

    def __init__(self, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        try:
            hyper_params = metadata[self.__class__.__name__][modifier_name]['pre_pollution_setting_id'][str(pre_pollution_setting_id)]['best_params'].copy()
            for key, value in hyper_params.items():
                if value == 'True':
                    hyper_params[key] = True
                elif value == 'False':
                    hyper_params[key] = False
                if value == 'None':
                    hyper_params[key] = None
        except KeyError:
            hyper_params = {'random_state': 42, 'loss': 'squared_error'}

        model = SGDClassifier(**hyper_params)
        super().__init__(self.__class__.__name__, metadata, model)

    def calculate_importance_score(self, encoder, row, ds_name, delta_x, delta_y, dirty_indexes_per_feature, dirty_record_index):
        md = self.metadata
        x = row.drop(md[ds_name]['target'])
        y = row[md[ds_name]['target']]
        # Convert x to dataframe
        x = pd.DataFrame(x).transpose()
        continuous_features_df = x.drop(md[ds_name]['categorical_cols'], axis=1)
        if encoder is not None:
            transformed_X_train = encoder.transform(x[md[ds_name]['categorical_cols']])
            x = np.hstack([transformed_X_train.toarray(), continuous_features_df.values])
        else:
            x = continuous_features_df.values
        if ds_name == 'cmc.data':
            theta = self.model.coef_
            theta = np.mean(theta, axis=0)
        else:
            theta = self.model.coef_.flatten()
        theta = np.reshape(theta, (-1, 1))
        if x.ndim == 1:
            x = x.reshape(1, -1)  # x is now 2D: (1, n_features)


        prediction = np.dot(x, theta) - y  # Resulting in shape (1, 1) because x is (1, n_features) and theta is (n_features, 1)
        gradient = (prediction * x).T  # Adjusted gradient computation; now gradient is (n_features, 1)

        # Initialize Mx matrix
        n_features = x.shape[1]  # Number of features from x
        Mx = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    # For diagonal elements, adjust the computation
                    Mx[i, i] = 2 * x[0, i] + (np.dot(theta.T, x.T) - theta[i, 0] * x[0, i] - y)[
                        0]  # Ensuring scalar assignment
                else:
                    # For off-diagonal elements
                    Mx[i, j] = theta[j, 0] * x[0, i]

        gradient = gradient.reshape(-1, 1)


        delta_xr = np.zeros((n_features,1))
        for feature in dirty_indexes_per_feature:
            if feature == md[ds_name]['target']:
                continue
            if dirty_record_index in dirty_indexes_per_feature[feature]:
                feature_index = row.columns.get_loc(feature)
                delta_xr[feature_index, 0] = delta_x[feature_index]
            else:
                feature_index = row.columns.get_loc(feature)
                delta_xr[feature_index, 0] = 0

        delta_yr = delta_y

        p_r = np.linalg.norm(gradient + np.dot(Mx, delta_xr))

        return p_r


class SGDClassifierExperimentLogisticRegression(ClassificationExperiment):

    def __init__(self, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        try:
            hyper_params = metadata[self.__class__.__name__][modifier_name]['pre_pollution_setting_id'][str(pre_pollution_setting_id)]['best_params'].copy()
            for key, value in hyper_params.items():
                if value == 'True':
                    hyper_params[key] = True
                elif value == 'False':
                    hyper_params[key] = False
                if value == 'None':
                    hyper_params[key] = None
        except KeyError:
            hyper_params = {'random_state': 42, 'loss': 'log'}

        model = SGDClassifier(**hyper_params)
        super().__init__(self.__class__.__name__, metadata, model)

    def calculate_importance_score(self, encoder, row, ds_name, delta_x, delta_y, dirty_indexes_per_feature, dirty_record_index):
        def sigmoid(z):
            if z >= 0:
                return 1 / (1 + np.exp(-z))
            else:
                # Handle negative z values to avoid overflow
                exp_z = np.exp(z)
                return exp_z / (1 + exp_z)

        md = self.metadata
        x = row.drop(md[ds_name]['target'])
        y = row[md[ds_name]['target']]
        x = pd.DataFrame(x).transpose()
        continuous_features_df = x.drop(md[ds_name]['categorical_cols'], axis=1)
        if encoder is not None:
            transformed_X_train = encoder.transform(x[md[ds_name]['categorical_cols']])
            x = np.hstack([transformed_X_train.toarray(), continuous_features_df.values])
        else:
            x = continuous_features_df.values
        if ds_name == 'cmc.data':
            theta = self.model.coef_
            theta = np.mean(theta, axis=0)
        else:
            theta = self.model.coef_.flatten()

        # Reshape x and theta for matrix operations
        x = x.reshape(1, -1)  # Ensures x is 2D with a single row
        theta = theta.reshape(-1, 1)  # Ensures theta is a column vector

        # Compute z and h_theta_x for logistic regression
        z = np.dot(x, theta)  # Results in a shape (1, 1)
        z = z.item()  # Converts z from a (1,1) array to a scalar
        h_theta_x = sigmoid(z)

        # Gradient calculation
        gradient = (h_theta_x - y) * x
        gradient = gradient.reshape(-1, 1)  # Ensure gradient is a column vector for consistency

        # Initialize Mx matrix
        n_features = x.shape[1]  # Correctly identifies the number of features
        Mx = np.zeros((n_features, n_features))

        # Calculate Mx according to provided formula
        for i in range(n_features):
            for j in range(n_features):
                sigmoid_term = h_theta_x * (1 - h_theta_x)
                if i == j:
                    theta_i_scalar = theta[i, 0].item()  # Ensures scalar value for theta[i]
                    xi_scalar = x[0, i]  # x is (1, n_features), so this is correct
                    Mx[i, i] = sigmoid_term * theta_i_scalar * xi_scalar + h_theta_x - y
                else:
                    theta_j_scalar = theta[j, 0].item()  # Ensures scalar value for theta[j]
                    Mx[i, j] = sigmoid_term * theta_j_scalar * x[0, i] + h_theta_x


        delta_xr = np.zeros((n_features,1))
        for feature in dirty_indexes_per_feature:
            if feature == md[ds_name]['target']:
                continue
            if dirty_record_index in dirty_indexes_per_feature[feature]:
                feature_index = row.columns.get_loc(feature)
                delta_xr[feature_index, 0] = delta_x[feature_index]
            else:
                feature_index = row.columns.get_loc(feature)
                delta_xr[feature_index, 0] = 0

        delta_yr = delta_y

        p_r = np.linalg.norm(gradient + np.dot(Mx, delta_xr))

        return p_r


def write_result_to_db(iteration, ds_name, ml_algorithm_str, error_type_name, pre_pollution_setting_id, db_engine, number_of_cleaned_cells, used_budget, f1_score):
    cleaning_results_df = DataFrame({'iteration': [iteration], 'pre_pollution_setting_id': [pre_pollution_setting_id],
                        'used_budget': [used_budget], 'f1_score': [f1_score], 'number_of_cleaned_cells': [number_of_cleaned_cells]})

    table_name = f'activeclean_results_{ds_name}_{ml_algorithm_str}_{error_type_name}'
    with open(f'{ROOT_DIR}/slurm/activeclean/RESULTS/{table_name}_{pre_pollution_setting_id}.csv', 'a') as f:
        if os.stat(f'{ROOT_DIR}/slurm/activeclean/RESULTS/{table_name}_{pre_pollution_setting_id}.csv').st_size == 0:
            cleaning_results_df.to_csv(f, header=True, index=False)
        else:
            cleaning_results_df.to_csv(f, header=False, index=False)


def main(ml_algorithm_class, error_type, ds_name: str, original_budget, metadata, database_engine, pre_pollution_setting_ids) -> None:

    table_name = f'activeclean_results_{ds_name}_{ml_algorithm_class.__name__}_{error_type.__name__}'
    #delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids)

    pre_pollution_df = load_pre_pollution_df(ds_name, error_type, database_engine)
    if pre_pollution_df.empty:
        warnings.warn(f'No pre pollution df found for {ds_name} and {error_type.__name__}. Stopping execution.'
                      f'Please run pre pollution first.', UserWarning)
        return None

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=pre_pollution_setting_ids)
    print(pre_pollution_settings)

    for pre_pollution_setting in pre_pollution_settings:
        pre_pollution_setting_id = pre_pollution_setting['pre_pollution_setting_id']
        if os.path.exists(f'{ROOT_DIR}/slurm/activeclean/RESULTS/{table_name}_{pre_pollution_setting_id}.csv'):
            os.remove(f'{ROOT_DIR}/slurm/activeclean/RESULTS/{table_name}_{pre_pollution_setting_id}.csv')

    for pollution_setting in pre_pollution_settings:
        print(f'Cleaning dataset for pollution setting {pollution_setting["pre_pollution_setting_id"]}.')
        # measure execution time for each pollution setting
        start_time = time.time()
        BUDGET = original_budget
        pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
        iteration = 0

        ml_algorithm = ml_algorithm_class(metadata, error_type.__name__, pre_pollution_setting_id)
        #pollution_setting = {'pre_pollution_setting_id': pre_pollution_setting_id, 'train': {'status': 0.01, 'credit_history': 0.02}, 'test': {'status': 0.01, 'credit_history': 0.02}}
        ap = ArtificialPollution2(metadata, str(database_engine.url), pre_pollution_setting_id, error_type, pre_pollution_df, ds_name, pollution_setting)

        train_df: DataFrame = ap.train_df
        test_df: DataFrame = ap.test_df
        train_df_polluted: DataFrame = ap.train_df_polluted
        test_df_polluted: DataFrame = ap.test_df_polluted

        label_encoder = LabelEncoder()
        train_df[metadata[ds_name]['target']] = label_encoder.fit_transform(train_df[metadata[ds_name]['target']])
        test_df[metadata[ds_name]['target']] = label_encoder.transform(test_df[metadata[ds_name]['target']])
        train_df_polluted[metadata[ds_name]['target']] = label_encoder.transform(train_df_polluted[metadata[ds_name]['target']])
        test_df_polluted[metadata[ds_name]['target']] = label_encoder.transform(test_df_polluted[metadata[ds_name]['target']])

        for col in metadata[ds_name]['categorical_cols']:
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)
            train_df_polluted[col] = train_df_polluted[col].astype(str)
            test_df_polluted[col] = test_df_polluted[col].astype(str)

        all_categories = []
        if metadata[ds_name]['categorical_cols']:
            if len(metadata[ds_name]['categorical_cols']) > 0:
                for column in metadata[ds_name]['categorical_cols']:
                    unique_values = list(train_df[column].unique())
                    unique_values.extend(list(test_df[column].unique()))
                    unique_values.extend(list(train_df_polluted[column].unique()))
                    unique_values.extend(list(test_df_polluted[column].unique()))
                    unique_values = list(set(unique_values))
                    all_categories.append(unique_values)
                encoder = OneHotEncoder(categories=all_categories, handle_unknown='ignore')
            else:
                encoder = None
        else:
            encoder = None

        dirty_indexes_per_feature_test, dirty_indexes_per_feature_train, dirty_records_test, dirty_records_train = find_dirty_records(
            test_df, test_df_polluted, train_df, train_df_polluted)

        clean_partial_test_df, clean_partial_train_df = find_originally_clean_records(dirty_records_test,
                                                                                      dirty_records_train, test_df,
                                                                                      train_df)
        if encoder is not None:
            encoder.fit(clean_partial_train_df[metadata[ds_name]['categorical_cols']])
        ml_algorithm.fit(encoder, ds_name, clean_partial_train_df)
        f1_score = ml_algorithm.predict(encoder, ds_name, clean_partial_test_df)
        print(f'Original F1 score: {f1_score} Budget: {BUDGET} Time: {time.time() - start_time}s')
        write_result_to_db(iteration, ds_name, ml_algorithm_str, error_type.__name__, pre_pollution_setting_id, database_engine, 0, original_budget-BUDGET, f1_score)
        iteration += 1

        cleaning_step_size = 0.01
        number_of_training_cells_per_iteration = cleaning_step_size * len(train_df)
        number_of_test_cells_per_iteration = cleaning_step_size * len(test_df)

        tracked_cleaned_records_df = pd.DataFrame(columns=train_df.columns)
        tracked_dirty_records_df = pd.DataFrame(columns=train_df.columns)


        cleaning_candidates_train_index = get_random_cleaning_candidates(dirty_indexes_per_feature_train, dirty_records_train,
                                                                         number_of_training_cells_per_iteration, train_df)
        cleaning_candidates_test_index = get_random_cleaning_candidates(dirty_indexes_per_feature_test, dirty_records_test,
                                                                        number_of_test_cells_per_iteration, test_df)

        new_clean_partial_train_df, temp_dirty_records, train_df_polluted = clean_data(cleaning_candidates_train_index, train_df, train_df_polluted, clean_partial_train_df)
        clean_partial_train_df = pd.concat([clean_partial_train_df, new_clean_partial_train_df])
        tracked_cleaned_records_df = pd.concat([tracked_cleaned_records_df, new_clean_partial_train_df])
        tracked_dirty_records_df = pd.concat([tracked_dirty_records_df, temp_dirty_records])

        new_clean_partial_test_df, temp_dirty_records, test_df_polluted = clean_data(cleaning_candidates_test_index, test_df, test_df_polluted, clean_partial_test_df)
        clean_partial_test_df = pd.concat([clean_partial_test_df, new_clean_partial_test_df])
        tracked_cleaned_records_df = pd.concat([tracked_cleaned_records_df, new_clean_partial_test_df])
        tracked_dirty_records_df = pd.concat([tracked_dirty_records_df, temp_dirty_records])
        BUDGET -= 1

        number_of_cleaned_cells = len([item for sublist in cleaning_candidates_train_index.values() for item in sublist])
        number_of_cleaned_cells += len([item for sublist in cleaning_candidates_test_index.values() for item in sublist])
        if number_of_cleaned_cells == 0:
            break

        clean_partial_train_df = clean_partial_train_df.sort_index()
        clean_partial_test_df = clean_partial_test_df.sort_index()
        if encoder is not None:
            encoder.fit(clean_partial_train_df[metadata[ds_name]['categorical_cols']])

        ml_algorithm.partial_fit(encoder, ds_name, clean_partial_train_df)
        temp_test_df = clean_partial_train_df.copy()
        temp_test_df = pd.concat([temp_test_df, test_df_polluted.loc[test_df_polluted.index.difference(temp_test_df.index)]])
        f1_score = ml_algorithm.predict(encoder, ds_name, temp_test_df)
        print(f'Iteration {iteration} F1 score: {f1_score} Budget: {BUDGET} Time: {time.time() - start_time}s Number of cleaned cells: {number_of_cleaned_cells}\n')
        write_result_to_db(iteration, ds_name, ml_algorithm_str, error_type.__name__, pre_pollution_setting_id, database_engine, number_of_cleaned_cells, original_budget-BUDGET, f1_score)
        iteration += 1

        index_list = []
        while BUDGET > 0:

            delta_x = calculate_delta_x(encoder, tracked_cleaned_records_df, tracked_dirty_records_df, len(train_df) + len(test_df))
            delta_y = calculate_delta_y(encoder, tracked_cleaned_records_df, tracked_dirty_records_df, len(train_df) + len(test_df))

            cleaning_candidates_train_index = get_importance_based_cleaning_candidates(encoder, ds_name, ml_algorithm, train_df_polluted, dirty_indexes_per_feature_train, dirty_records_train, number_of_training_cells_per_iteration, train_df, delta_x, delta_y)
            cleaning_candidates_test_index = get_importance_based_cleaning_candidates(encoder, ds_name, ml_algorithm, test_df_polluted, dirty_indexes_per_feature_test, dirty_records_test, number_of_test_cells_per_iteration, test_df, delta_x, delta_y)
            index_list.append(cleaning_candidates_train_index)
            new_clean_partial_train_df, temp_dirty_records, train_df_polluted = clean_data(cleaning_candidates_train_index, train_df,
                                                                        train_df_polluted, clean_partial_train_df)
            clean_partial_train_df = pd.concat([clean_partial_train_df, new_clean_partial_train_df])
            tracked_cleaned_records_df = pd.concat([tracked_cleaned_records_df, new_clean_partial_train_df])
            tracked_dirty_records_df = pd.concat([tracked_dirty_records_df, temp_dirty_records])

            new_clean_partial_test_df, temp_dirty_records, test_df_polluted = clean_data(cleaning_candidates_test_index, test_df,
                                                                       test_df_polluted, clean_partial_test_df)
            clean_partial_test_df = pd.concat([clean_partial_test_df, new_clean_partial_test_df])
            tracked_cleaned_records_df = pd.concat([tracked_cleaned_records_df, new_clean_partial_test_df])
            tracked_dirty_records_df = pd.concat([tracked_dirty_records_df, temp_dirty_records])
            BUDGET -= 1

            if encoder is not None:
                encoder.fit(clean_partial_train_df[metadata[ds_name]['categorical_cols']])

            number_of_cleaned_cells = len([item for sublist in cleaning_candidates_train_index.values() for item in sublist])
            number_of_cleaned_cells += len([item for sublist in cleaning_candidates_test_index.values() for item in sublist])

            if number_of_cleaned_cells == 0:
                break

            clean_partial_train_df = clean_partial_train_df.sort_index()
            clean_partial_test_df = clean_partial_test_df.sort_index()
            ml_algorithm.partial_fit(encoder, ds_name, clean_partial_train_df)
            temp_test_df = clean_partial_train_df.copy()
            temp_test_df = pd.concat([temp_test_df, test_df_polluted.loc[test_df_polluted.index.difference(temp_test_df.index)]])
            f1_score = ml_algorithm.predict(encoder, ds_name, temp_test_df)

            dirty_indexes_per_feature_test, dirty_indexes_per_feature_train, dirty_records_test, dirty_records_train = find_dirty_records(
                test_df, test_df_polluted, train_df, train_df_polluted)

            write_result_to_db(iteration, ds_name, ml_algorithm_str, error_type.__name__, pre_pollution_setting_id, database_engine, number_of_cleaned_cells, original_budget-BUDGET, f1_score)
            print(f'Iteration {iteration} F1 score: {f1_score} Budget: {BUDGET} Time: {time.time() - start_time}s Number of cleaned cells: {number_of_cleaned_cells}\n')
            iteration += 1
        print(f'Finished pollution setting {pre_pollution_setting_id} in {time.time() - start_time} seconds.')

        ml_algorithm = ml_algorithm_class(metadata, error_type.__name__, pre_pollution_setting_id)
        if encoder is not None:
            encoder.fit(train_df[metadata[ds_name]['categorical_cols']])
        ml_algorithm.fit(encoder, ds_name, train_df)
        f1_score = ml_algorithm.predict(encoder, ds_name, test_df)
        print(f'F1 score for fully cleaned dataset: {f1_score}')


def find_originally_clean_records(dirty_records_test, dirty_records_train, test_df, train_df):
    originally_clean_records_train = [index for index in train_df.index if index not in dirty_records_train]
    originally_clean_records_test = [index for index in test_df.index if index not in dirty_records_test]
    clean_partial_train_df = train_df.loc[originally_clean_records_train]
    clean_partial_test_df = test_df.loc[originally_clean_records_test]
    return clean_partial_test_df, clean_partial_train_df


def find_dirty_records(test_df, test_df_polluted, train_df, train_df_polluted):
    dirty_indexes_per_feature_train = {}
    dirty_indexes_per_feature_test = {}
    dirty_records_train = []
    dirty_records_test = []
    for feature in train_df.columns:
        mismatches: Series = train_df_polluted[feature] != train_df[feature]
        dirty_indexes_per_feature_train[feature] = mismatches[mismatches].index.tolist()

        mismatches: Series = test_df_polluted[feature] != test_df[feature]
        dirty_indexes_per_feature_test[feature] = mismatches[mismatches].index.tolist()

        dirty_records_train.extend(dirty_indexes_per_feature_train[feature])
        dirty_records_test.extend(dirty_indexes_per_feature_test[feature])
    dirty_records_train = list(set(dirty_records_train))
    dirty_records_test = list(set(dirty_records_test))
    return dirty_indexes_per_feature_test, dirty_indexes_per_feature_train, dirty_records_test, dirty_records_train


def calculate_delta_x(encoder, cleaned_records_df, dirty_records_df, N):
    X_train = cleaned_records_df.drop(metadata[ds_name]['target'], axis=1)
    continuous_features_df = X_train.drop(metadata[ds_name]['categorical_cols'], axis=1)
    if encoder is not None:
        transformed_X_train = encoder.transform(X_train[metadata[ds_name]['categorical_cols']])
        cleaned_records_prepared = np.hstack([transformed_X_train.toarray(), continuous_features_df])
    else:
        cleaned_records_prepared = X_train

    X_train = dirty_records_df.drop(metadata[ds_name]['target'], axis=1)
    continuous_features_df = X_train.drop(metadata[ds_name]['categorical_cols'], axis=1)
    if encoder is not None:
        transformed_X_train = encoder.transform(X_train[metadata[ds_name]['categorical_cols']])
        dirty_records_prepared = np.hstack([transformed_X_train.toarray(), continuous_features_df])
    else:
        dirty_records_prepared = X_train

    differences = dirty_records_prepared - cleaned_records_prepared

    # Uniform sampling probability
    p_j = 1 / len(cleaned_records_df)
    delta_x = (1/(N * len(cleaned_records_prepared))) * np.sum(differences, axis=0) * (1 / p_j)
    return delta_x

def calculate_delta_y(encoder, cleaned_records_df, dirty_records_df, N):
    # create differences ndarray of the shape of cleaned_records_df dataframe
    cleaned_labels = cleaned_records_df[metadata[ds_name]['target']].values
    dirty_labels = dirty_records_df[metadata[ds_name]['target']].values

    cleaned_labels = np.array(cleaned_labels)
    dirty_labels = np.array(dirty_labels)

    # initialize vector zero vector
    differences = np.ndarray(shape=(len(cleaned_labels),), dtype=float)
    differences.fill(0)

    # Uniform sampling probability
    p_j = 1 / N

    # Calculate Delta y, weighted by the sampling probability
    delta_y = (1/(N * len(cleaned_records_df))) * np.sum(differences) * (1 / p_j)

    delta_y = 0.0
    return delta_y


def clean_data(cleaning_candidates_index, df_clean, df_polluted, clean_partial_df):
    dtype_dict = {col: dtype for col, dtype in zip(clean_partial_df.columns, clean_partial_df.dtypes)}
    new_clean_partial_df = pd.DataFrame(columns=clean_partial_df.columns).astype(dtype_dict)
    dirty_partial_df = pd.DataFrame(columns=clean_partial_df.columns).astype(dtype_dict)
    source_dtypes = clean_partial_df.dtypes
    for record_index in cleaning_candidates_index:
        temp_df = df_polluted.loc[record_index].to_frame().transpose().copy()
        temp_df = temp_df.astype(source_dtypes)
        new_clean_partial_df = pd.concat([new_clean_partial_df, temp_df])
        dirty_partial_df = pd.concat([dirty_partial_df, temp_df])
        for feature in cleaning_candidates_index[record_index]:
            new_clean_partial_df.at[record_index, feature] = df_clean.at[record_index, feature]
            df_polluted.at[record_index, feature] = df_clean.at[record_index, feature]
    return new_clean_partial_df, dirty_partial_df, df_polluted


def get_importance_based_cleaning_candidates(encoder, ds_name, ml_algorithm, polluted_df, dirty_indexes_per_feature, dirty_records, number_of_cells_per_iteration, clean_df, delta_x, delta_y):
    importance_scores = {}
    for dirty_record_index in dirty_records:
        row = polluted_df.loc[dirty_record_index]
        row.columns = clean_df.columns
        importance_scores[dirty_record_index] = ml_algorithm.calculate_importance_score(encoder, row, ds_name, delta_x, delta_y, dirty_indexes_per_feature, dirty_record_index)

    cleaning_candidates_index = {}
    found_dirty_cells_counter = 0
    while found_dirty_cells_counter < number_of_cells_per_iteration:
        if len(dirty_records) == 0 or len(importance_scores) == 0:
            break
        record_index_to_clean = max(importance_scores, key=importance_scores.get)
        del importance_scores[record_index_to_clean]
        dirty_features = [feature for feature in clean_df.columns if record_index_to_clean in dirty_indexes_per_feature[feature]]
        cleaning_candidates_index[record_index_to_clean] = dirty_features
        found_dirty_cells_counter += len(dirty_features)
    return cleaning_candidates_index


def get_random_cleaning_candidates(dirty_indexes_per_feature, dirty_records, number_of_cells_per_iteration, clean_df):
    np.random.seed(42)
    cleaning_candidates_index = {}
    found_dirty_cells_counter = 0
    while found_dirty_cells_counter < number_of_cells_per_iteration:
        if len(dirty_records) == 0:
            break
        record_index_to_clean = dirty_records.pop()
        dirty_features = [feature for feature in clean_df.columns if record_index_to_clean in dirty_indexes_per_feature[feature]]
        cleaning_candidates_index[record_index_to_clean] = dirty_features
        found_dirty_cells_counter += len(dirty_features)
    return cleaning_candidates_index


if __name__ == "__main__":

    ml_algorithms = {'SGDClassifierExperimentSVM': SGDClassifierExperimentSVM,
                     'SGDClassifierExperimentLinearRegression': SGDClassifierExperimentLinearRegression,
                     'SGDClassifierExperimentLogisticRegression': SGDClassifierExperimentLogisticRegression}

    error_types = {'MissingValuesModifier': MissingValuesModifier,
                   'CategoricalShiftModifier': CategoricalShiftModifier,
                   'ScalingModifier': ScalingModifier,
                   'GaussianNoiseModifier': GaussianNoiseModifier}

    parser = argparse.ArgumentParser()
    parser.add_argument('--ml_algorithm', default='SGDClassifierExperimentLinearRegression', type=str, help='Set the ML algorithm to use for the experiment.')
    parser.add_argument('--error_type', default='ScalingModifier', type=str, help='Set the error type to use for the experiment.')
    parser.add_argument('--dataset', default='Airbnb', type=str, help='Set the dataset to use for the experiment.')
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

    database_engine = create_engine(database_url, echo=False, connect_args={'timeout': 1500})

    try:
        metadata = load_json(open(args.metadata, 'r'))
    except FileNotFoundError:
        print(f'Could not find metadata.json file at {args.metadata}.')
        quit()

    main(chosen_ml_algorithm, chosen_error_type, ds_name, budget, metadata, database_engine, pre_pollution_setting_ids)
