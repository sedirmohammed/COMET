from json import load as load_json

from sklearn.base import clone
from classification.experiments import *
from config.definitions import ROOT_DIR
import itertools
from classification.utils.artifical_pollution import *
from classification.utils.util import get_pre_pollution_settings, load_pre_pollution_df
from classification.utils.DatasetModifier import *
import json
from classification.pre_pollution import generate_multi_error_data


class StaticRandomHyperParameterSearchExperiment:

    def __init__(self, df_train, y_train, df_test, y_test, model, param_grid, n_samples=10):
        self.df_train = df_train.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.y_test = y_test.copy(deep=True)
        self.model = model
        self.param_grid = param_grid
        self.n_samples = n_samples

    def hyperparameter_search(self):
        keys, values = zip(*self.param_grid.items())
        hyper_param_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print('Generated all possible hyper param configurations')
        rng = np.random.RandomState(42)
        hyper_param_configurations = rng.choice(hyper_param_configurations, size=self.n_samples, replace=False)
        best_params = None
        best_f1 = 0.0
        hyper_param_configurations_results = Parallel(n_jobs=10)(
            delayed(self.parallel_hyper_param_configuration_calculation)(hyper_param_configuration)
            for hyper_param_configuration in hyper_param_configurations)

        for hyper_param_configuration_result in hyper_param_configurations_results:
            current_f1 = hyper_param_configuration_result['f1']
            if current_f1 is None:
                continue
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = hyper_param_configuration_result['hyper_param_configuration']

        return best_params

    def parallel_hyper_param_configuration_calculation(self, hyper_param_configuration: dict):
        try:
            local_model = clone(self.model)
            local_model.set_params(**hyper_param_configuration)
            local_model.fit(self.df_train, self.y_train)
            y_pred = local_model.predict(self.df_test)
            current_f1 = float(classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score'])
        except Exception as e:
            print(e)
            return {'hyper_param_configuration': hyper_param_configuration, 'f1': None}
        return {'hyper_param_configuration': hyper_param_configuration, 'f1': current_f1}


class SupportVectorMachineExperiment:
    """
    This model needs the categorical columns from the metadata to target-encode them.
    z-score standardisation should be used for scaling the features, if wanted.
    """

    def __init__(self, df_train, df_test, metadata) -> None:
        self.model = SVC(random_state=42)
        self.metadata = metadata
        df_train, df_test = target_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        y_train = df_train[metadata['target']]
        y_test = df_test[metadata['target']]

        columns_to_drop = [metadata['target']]
        if 'index' in df_train.keys():
            columns_to_drop.append('index')

        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)

        scaler = StandardScaler()
        df_train = pd.DataFrame(scaler.fit_transform(df_train), index=df_train.index, columns=df_train.columns)
        df_test = pd.DataFrame(scaler.transform(df_test), index=df_test.index, columns=df_test.columns)

        self.df_train = df_train
        self.df_test = df_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = None

    def run(self):
        self.model.fit(self.df_train, self.y_train)
        y_pred = self.model.predict(self.df_test)
        return classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    def hyperparameter_search(self, n_samples=10):
        """
        This method does a hyperparameter search for the model.
        :param n_samples: The number of samples to use for the random search
        :return
        """
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [1, 2, 3, 4, 5, 6],
            'coef0': [0.0, 0.1, 0.5, 1.0],
            'shrinking': [True, False],
            'probability': [False],
            'tol': [0.001, 0.0001, 0.00001],
            'cache_size': [200, 500, 1000],
            'class_weight': [None, 'balanced'],
            'max_iter': [100, 500, 1000, 2000, 5000],
            'decision_function_shape': ['ovo', 'ovr'],
            'break_ties': [True, False],
            'random_state': [42]
        }
        self.best_params = StaticRandomHyperParameterSearchExperiment(self.df_train, self.y_train, self.df_test, self.y_test, self.model, param_grid, n_samples=n_samples).hyperparameter_search()
        self.model = SVC(**self.best_params)


class KNeighborsExperiment:
    def __init__(self, df_train, df_test, metadata) -> None:
        self.model = KNeighborsClassifier()
        self.metadata = metadata
        df_train, df_test = target_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        y_train = df_train[metadata['target']]
        y_test = df_test[metadata['target']]

        columns_to_drop = [metadata['target']]
        if 'index' in df_train.keys():
            columns_to_drop.append('index')

        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)

        self.df_train = df_train
        self.df_test = df_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = None

    def run(self):
        self.model.fit(self.df_train, self.y_train)
        y_pred = self.model.predict(self.df_test)
        return classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    def hyperparameter_search(self, n_samples=10):

        param_grid = {
            'n_neighbors': [1, 2, 3, 4, 5],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 30, 40],
            'p': [1, 2],
            'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
            'n_jobs': [1]
        }
        self.best_params = StaticRandomHyperParameterSearchExperiment(self.df_train, self.y_train, self.df_test, self.y_test, self.model, param_grid, n_samples=n_samples).hyperparameter_search()
        self.model = KNeighborsClassifier(**self.best_params)


class GradientBoostingExperiment:
    def __init__(self, df_train, df_test, metadata) -> None:
        self.model = GradientBoostingClassifier(random_state=42)
        self.metadata = metadata
        df_train, df_test = target_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        y_train = df_train[metadata['target']]
        y_test = df_test[metadata['target']]

        columns_to_drop = [metadata['target']]
        if 'index' in df_train.keys():
            columns_to_drop.append('index')

        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)

        self.df_train = df_train
        self.df_test = df_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = None

    def run(self):
        self.model.fit(self.df_train, self.y_train)
        y_pred = self.model.predict(self.df_test)
        return classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    def hyperparameter_search(self, n_samples=10):

        param_grid = {
            'loss': ['log_loss', 'exponential'],
            'learning_rate': [0.01, 0.1, 1.0],
            'n_estimators': [100, 1000],
            'subsample': [0.1, 0.5, 1.0],
            'criterion': ['friedman_mse', 'squared_error', 'mae'],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 2, 4],
            'min_weight_fraction_leaf': [0.0, 0.1, 0.5],
            'max_depth': [3, 10, 100],
            'min_impurity_decrease': [0.0, 0.1, 0.5],
            'init': [None, 'zero'],
            'random_state': [42],
            'max_features': [1.0, 'sqrt', 'log2', None],
            'max_leaf_nodes': [None, 2, 20],
            'warm_start': [False],
            'validation_fraction': [0.1, 0.2, 0.5],
            'n_iter_no_change': [None, 1, 10],
            'tol': [0.0001, 0.01],
            'ccp_alpha': [0.0, 1.0]
        }
        self.best_params = StaticRandomHyperParameterSearchExperiment(self.df_train, self.y_train, self.df_test, self.y_test, self.model, param_grid, n_samples=n_samples).hyperparameter_search()
        self.model = GradientBoostingClassifier(**self.best_params)


class MultilayerPerceptronExperiment:
    def __init__(self, df_train, df_test, metadata) -> None:
        self.model = MLPClassifier(random_state=42)
        self.metadata = metadata
        df_train, df_test = target_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        y_train = df_train[metadata['target']]
        y_test = df_test[metadata['target']]

        columns_to_drop = [metadata['target']]
        if 'index' in df_train.keys():
            columns_to_drop.append('index')

        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)

        self.df_train = df_train
        self.df_test = df_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = None

    def run(self):
        self.model.fit(self.df_train, self.y_train)
        y_pred = self.model.predict(self.df_test)
        return classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    def hyperparameter_search(self, n_samples=10):

        param_grid = {
            'hidden_layer_sizes': [(8,), (16,), (32,), (8, 8,), (16, 16,), (32, 32,), (8, 8, 8, 8), (16, 16, 16, 16), (32, 32, 32, 32)],
            'activation': ['identity', 'tanh', 'relu'],
            'solver': ['adam'],
            'batch_size': [200, 500, 1000],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'power_t': [0.5, 0.1, 0.01],
            'max_iter': [100, 200, 500],
            'shuffle': [True, False],
            'random_state': [42],
            'tol': [0.0001, 0.01, 0.1],
            'warm_start': [False, True],
            'momentum': [0.9, 0.5],
            'nesterovs_momentum': [True, False],
            'early_stopping': [True, False],
            'validation_fraction': [0.01, 0.1, 0.2],
            'beta_1': [0.9, 0.1],
            'beta_2': [0.999, 0.1],
            'epsilon': [1e-07, 1e-08, 1e-09],
            'n_iter_no_change': [10, 20, 50]
        }
        self.best_params = StaticRandomHyperParameterSearchExperiment(self.df_train, self.y_train, self.df_test, self.y_test, self.model, param_grid, n_samples=n_samples).hyperparameter_search()
        self.model = MLPClassifier(**self.best_params)

class SGDClassifierExperimentLinearRegression:
    def __init__(self, df_train, df_test, metadata) -> None:
        self.model = SGDClassifier(random_state=42, loss='squared_error')
        self.metadata = metadata
        df_train, df_test = target_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        y_train = df_train[metadata['target']]
        y_test = df_test[metadata['target']]

        columns_to_drop = [metadata['target']]
        if 'index' in df_train.keys():
            columns_to_drop.append('index')

        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)

        self.df_train = df_train
        self.df_test = df_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = None

    def run(self):
        self.model.fit(self.df_train, self.y_train)
        y_pred = self.model.predict(self.df_test)
        return classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    def hyperparameter_search(self, n_samples=10):

            param_grid = {
                'loss': ['squared_error'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001, 0.01],
                'l1_ratio': [0.15, 0.1, 0.5],
                'fit_intercept': [True, False],
                'max_iter': [1000, 2000, 5000],
                'tol': [0.0001, 0.001, 0.01],
                'shuffle': [True, False],
                'epsilon': [0.1, 0.01, 0.001],
                'random_state': [42]
            }
            self.best_params = StaticRandomHyperParameterSearchExperiment(self.df_train, self.y_train, self.df_test, self.y_test, self.model, param_grid, n_samples=n_samples).hyperparameter_search()
            self.model = SGDClassifier(**self.best_params)

class SGDClassifierExperimentLogisticRegression:
    def __init__(self, df_train, df_test, metadata) -> None:
        self.model = SGDClassifier(random_state=42, loss='log')
        self.metadata = metadata
        df_train, df_test = target_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        y_train = df_train[metadata['target']]
        y_test = df_test[metadata['target']]

        columns_to_drop = [metadata['target']]
        if 'index' in df_train.keys():
            columns_to_drop.append('index')

        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)

        self.df_train = df_train
        self.df_test = df_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = None

    def run(self):
        self.model.fit(self.df_train, self.y_train)
        y_pred = self.model.predict(self.df_test)
        return classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    def hyperparameter_search(self, n_samples=10):

            param_grid = {
                'loss': ['log'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001, 0.01],
                'l1_ratio': [0.15, 0.1, 0.5],
                'fit_intercept': [True, False],
                'max_iter': [1000, 2000, 5000],
                'tol': [0.0001, 0.001, 0.01],
                'shuffle': [True, False],
                'epsilon': [0.1, 0.01, 0.001],
                'random_state': [42]
            }
            self.best_params = StaticRandomHyperParameterSearchExperiment(self.df_train, self.y_train, self.df_test, self.y_test, self.model, param_grid, n_samples=n_samples).hyperparameter_search()
            self.model = SGDClassifier(**self.best_params)


class SGDClassifierExperimentSVM:
    def __init__(self, df_train, df_test, metadata) -> None:
        self.model = SGDClassifier(random_state=42, loss='hinge')
        self.metadata = metadata
        df_train, df_test = target_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        y_train = df_train[metadata['target']]
        y_test = df_test[metadata['target']]

        columns_to_drop = [metadata['target']]
        if 'index' in df_train.keys():
            columns_to_drop.append('index')

        df_train = df_train.drop(columns=columns_to_drop)
        df_test = df_test.drop(columns=columns_to_drop)

        self.df_train = df_train
        self.df_test = df_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_params = None

    def run(self):
        self.model.fit(self.df_train, self.y_train)
        y_pred = self.model.predict(self.df_test)
        return classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    def hyperparameter_search(self, n_samples=10):
        param_grid = {
            'loss': ['hinge'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'l1_ratio': [0.15, 0.1, 0.5],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000, 5000],
            'tol': [0.0001, 0.001, 0.01],
            'shuffle': [True, False],
            'epsilon': [0.1, 0.01, 0.001],
            'random_state': [42]
        }
        self.best_params = StaticRandomHyperParameterSearchExperiment(self.df_train, self.y_train, self.df_test, self.y_test, self.model, param_grid, n_samples=n_samples).hyperparameter_search()
        self.model = SGDClassifier(**self.best_params)


metadata_path = os.path.join(ROOT_DIR, 'metadata.json')
metadata = load_json(open(metadata_path, 'r'))


best_params: dict = {}
f1_scores: dict = {}
DS_NAMES = ['EEG.arff', 'SouthGermanCredit.csv', 'TelcoCustomerChurn.csv', 'cmc.data']
ERROR_TYPES = [CategoricalShiftModifier, MissingValuesModifier, GaussianNoiseModifier, ScalingModifier]
EXPERIMENTS = [MultilayerPerceptronExperiment, KNeighborsExperiment, GradientBoostingExperiment]
database_url = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
database_engine = create_engine(database_url, echo=True, connect_args={'timeout': 1000})

for ds_name in DS_NAMES:
    print(f'Running {ds_name}')

    print('Start preprocessing')
    best_params[ds_name] = {}
    f1_scores[ds_name] = {}

    best_params[ds_name]['multi_error'] = {}
    f1_scores[ds_name]['multi_error'] = {}

    pre_pollution_settings = get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=[1, 2, 3])
    for pollution_setting in pre_pollution_settings:
        print(f'Running {pollution_setting}')
        pre_pollution_setting_id = pollution_setting['pre_pollution_setting_id']
        best_params[ds_name]['multi_error'][str(pre_pollution_setting_id)] = {}
        f1_scores[ds_name]['multi_error'][str(pre_pollution_setting_id)] = {}

        error_map_test_df, error_map_train_df, test_df, test_df_polluted, train_df, train_df_polluted = generate_multi_error_data(
            database_engine, ds_name, ERROR_TYPES, metadata, pollution_setting, pre_pollution_setting_id)

        print('Finished preprocessing')

        for experiment in EXPERIMENTS:
            print(f'Running {experiment.__name__}')
            exp = experiment(train_df_polluted, test_df_polluted, metadata[ds_name])
            exp.hyperparameter_search(n_samples=10)
            best_params[ds_name]['multi_error'][str(pre_pollution_setting_id)][exp.__class__.__name__] = exp.best_params
            f1_scores[ds_name]['multi_error'][str(pre_pollution_setting_id)][exp.__class__.__name__] = exp.run()
            print(f'Finished {exp.__class__.__name__} for {ds_name}')
            print(f'F1 score: {f1_scores[ds_name]["multi_error"][str(pre_pollution_setting_id)][exp.__class__.__name__]}')
            print(f'Best parameters: {best_params[ds_name]["multi_error"][str(pre_pollution_setting_id)][exp.__class__.__name__]}')
            print('---------------------------------')

print('\n\n\n')

new_metadata = metadata.copy()


def add_missing_keys():
    if ds_name not in new_metadata.keys():
        new_metadata[ds_name] = {}
        new_metadata[ds_name][experiment] = {}
        new_metadata[ds_name][experiment][error_type] = {}
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'] = {}
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'][pre_pollution_setting_id] = {}
    if experiment not in new_metadata[ds_name].keys():
        new_metadata[ds_name][experiment] = {}
        new_metadata[ds_name][experiment][error_type] = {}
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'] = {}
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'][pre_pollution_setting_id] = {}
    if error_type not in new_metadata[ds_name][experiment].keys():
        new_metadata[ds_name][experiment][error_type] = {}
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'] = {}
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'][pre_pollution_setting_id] = {}
    if 'pre_pollution_setting_id' not in new_metadata[ds_name][experiment][error_type].keys():
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'] = {}
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'][pre_pollution_setting_id] = {}
    if pre_pollution_setting_id not in new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'].keys():
        new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'][pre_pollution_setting_id] = {}


for ds_name in best_params.keys():
    for error_type in best_params[ds_name].keys():
        for pre_pollution_setting_id in best_params[ds_name][error_type].keys():
            pre_pollution_setting_id = str(pre_pollution_setting_id)
            for experiment in best_params[ds_name][error_type][pre_pollution_setting_id].keys():
                print(f'BEST PARAMS FOR {ds_name} {error_type} {pre_pollution_setting_id} {experiment}')
                print(best_params[ds_name][error_type][pre_pollution_setting_id][experiment])
                print(f'F1 score for {ds_name} {error_type} {pre_pollution_setting_id} {experiment}')
                print(f'{f1_scores[ds_name][error_type][pre_pollution_setting_id][experiment]}')
                print('---------------------------------')
                print('')

                for key in best_params[ds_name][error_type][pre_pollution_setting_id][experiment].keys():
                    # check if value is numeric
                    if not isinstance(best_params[ds_name][error_type][pre_pollution_setting_id][experiment][key], (int, float, list)):
                        best_params[ds_name][error_type][pre_pollution_setting_id][experiment][key] = str(best_params[ds_name][error_type][pre_pollution_setting_id][experiment][key])

                add_missing_keys()
                new_metadata[ds_name][experiment][error_type]['pre_pollution_setting_id'][pre_pollution_setting_id]['best_params'] = best_params[ds_name][error_type][pre_pollution_setting_id][experiment]

print(new_metadata)
with open(f'{ROOT_DIR}/metadata.json', 'w') as f:
    json.dump(new_metadata, f, indent=4)

print('Finished writing to metadata.json')
