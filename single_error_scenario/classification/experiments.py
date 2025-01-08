import ast
from abc import ABC
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from experiment import Experiment
from util import target_encode, one_hot_encode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from logging import info as logging_info
import shap
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings


class ClassificationExperiment(Experiment, ABC):
    def __init__(self, name, df_train, df_test, model, target, scaler=None):
        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.target = target
        self.scaler = scaler
        super().__init__(name, df_train, df_test, model)

    def __get_features_importance(self, data: DataFrame):
        features_importance = {'global': {}, 'local': {}}

        explainer = shap.Explainer(self.model.predict, data, seed=42)
        shap_values = explainer(data)

        vals = shap_values.values.mean(0)
        feature_importance = pd.DataFrame(list(zip(data.columns, vals)),
                                          columns=['features', 'feature_importance_vals'])
        features_importance['global'] = dict(feature_importance.values)
        print(feature_importance)
        return features_importance

    def run(self, polluter, scenario_name, explain=True):

        # Extract target and drop unnecessary columns
        X_train = self.df_train.drop(columns=[self.target])
        X_test = self.df_test.drop(columns=[self.target])

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(self.df_train[self.target])
        y_test = label_encoder.fit_transform(self.df_test[self.target])

        if self.scaler:
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(self.scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        test_result = classification_report(y_test, y_pred, output_dict=True)

        if explain:
            logging_info(f'Calculating features importance')
            features_importance = {'fi_over_test': self.__get_features_importance(data=X_test),
                                   'fi_over_train': []}
        else:
            features_importance = []

        return {self.name: {'scoring': test_result, 'feature_importances': features_importance}}

    @classmethod
    def get_classname(cls):
        return cls.__name__


class LogRegExperiment(ClassificationExperiment):
    name = 'Logistic Regression Classification'

    def __init__(self, df_train, df_test, metadata) -> None:
        """
        This model needs the categorical columns from the metadata to target-encode them.
        Min-Max-Normalisation should be used for scaling the features, if wanted.
        """
        model = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=2000, random_state=42, n_jobs=-1)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('Logistic Regression Classification', df_train, df_test, model, metadata['target'])


class KNeighborsExperiment(ClassificationExperiment):
    name = 'k-Nearest Neighbors Classification'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
            hyper_params = {}
        hyper_params['n_jobs'] = -1
        model = KNeighborsClassifier(**hyper_params)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('k-Nearest Neighbors Classification', df_train, df_test, model, metadata['target'],
                         MinMaxScaler())


def check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id):
    if modifier_name is None:
        warnings.warn('No modifier name was given. Using default hyperparameters.', UserWarning)
    if pre_pollution_setting_id is None:
        warnings.warn('No pre pollution setting id was given. Using default hyperparameters.', UserWarning)


class DecisionTreeExperiment(ClassificationExperiment):
    """
    Tree-based algorithms does not need any feature scaling.
    """

    name = 'Decision Tree Classification'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
            hyper_params = {'random_state': 42, 'min_samples_leaf': 1}

        model = DecisionTreeClassifier(**hyper_params)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('Decision Tree Classification', df_train, df_test, model, metadata['target'])


class MultilayerPerceptronExperiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to target-encode them.
    Min-Max-Normalisation should be used for scaling the features, if wanted.
    """
    name = 'Multilayer Perceptron Classification'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

        try:
            hyper_params = metadata[self.__class__.__name__][modifier_name]['pre_pollution_setting_id'][str(pre_pollution_setting_id)]['best_params'].copy()
            for key, value in hyper_params.items():
                if value == 'True':
                    hyper_params[key] = True
                elif value == 'False':
                    hyper_params[key] = False
                if key == 'hidden_layer_sizes':
                    hyper_params[key] = ast.literal_eval(hyper_params[key])
                if value == 'None':
                    hyper_params[key] = None
        except KeyError:
            hyper_params = {'random_state': 42, 'max_iter': 1000}

        model = MLPClassifier(**hyper_params)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('Multilayer Perceptron Classification', df_train, df_test, model, metadata['target'],
                         StandardScaler())


class SupportVectorMachineExperiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to target-encode them.
    z-score standardisation should be used for scaling the features, if wanted.
    """

    name = 'Support Vector Machine Classification'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
            hyper_params = {'random_state': 42, 'kernel': 'linear'}

        model = SVC(**hyper_params)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('Support Vector Machine Classification', df_train, df_test, model, metadata['target'],
                         StandardScaler())


class RandomForrestExperiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to target-encode them.
    z-score standardisation should be used for scaling the features, if wanted.
    """

    name = 'Random Forest Classification'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
            hyper_params = {'random_state': 42, 'kernel': 'linear'}
        model = RandomForestClassifier(**hyper_params)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('Random Forest Classification', df_train, df_test, model, metadata['target'],
                         StandardScaler())


class GradientBoostingExperiment(ClassificationExperiment):
    """
    This model needs the categorical columns from the metadata to target-encode them.
    z-score standardisation should be used for scaling the features, if wanted.
    """

    name = 'Gradient Boosting Classification'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
            hyper_params = {'random_state': 42}

        model = GradientBoostingClassifier(**hyper_params)
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('Gradient Boosting Classification', df_train, df_test, model, metadata['target'],
                         StandardScaler())


class SGDClassifierExperimentSVM(ClassificationExperiment):
    name = 'SGDClassifierExperimentSVM'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('SGDClassifierExperimentSVM', df_train, df_test, model, metadata['target'],
                         StandardScaler())


class SGDClassifierExperimentLogisticRegression(ClassificationExperiment):
    name = 'SGDClassifierExperimentLogisticRegression'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('SGDClassifierExperimentLogisticRegression', df_train, df_test, model, metadata['target'],
                         StandardScaler())


class SGDClassifierExperimentLinearRegression(ClassificationExperiment):
    name = 'SGDClassifierExperimentLinearRegression'

    def __init__(self, df_train, df_test, metadata, modifier_name=None, pre_pollution_setting_id=None) -> None:
        check_modifier_and_pre_pollution_setting_id(modifier_name, pre_pollution_setting_id)

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
        df_train, df_test = one_hot_encode(df_train, df_test, metadata['categorical_cols'], metadata['target'])

        super().__init__('SGDClassifierExperimentLinearRegression', df_train, df_test, model, metadata['target'],
                         StandardScaler())
