from joblib import Parallel, delayed

from classification.utils.regression_model import *
from pandas import DataFrame
import numpy as np
from classification.utils.artifical_pollution_new import PollutedFeature
from typing import Dict, List
from tqdm import tqdm


class ExperimentRunner:
    def __init__(self, all_polluted_features, ml_algorithm,  **kwargs):
        self.all_polluted_features_results: Dict[str, List[PollutedFeature]] = all_polluted_features
        self.ml_algorithm = ml_algorithm
        self.metadata = kwargs.get('metadata', None)
        self.ds_name = kwargs.get('ds_name', None)
        self.error_type = kwargs.get('error_type', None)
        self.pre_pollution_setting_id = kwargs.get('pre_pollution_setting_id', None)
        self.experiment_results = {}

    def run(self):
        # iterate over the polluted dfs keys and run the experiment
        counter = 0
        for feature in tqdm(self.all_polluted_features_results, desc="Processing features"):
            # if counter == 1:
            #     break
            # counter += 1
            polluted_feature_results = self.all_polluted_features_results[feature]
            results_df = pd.DataFrame(columns=['pollution_level', 'real_f1', 'predicted_f1'])
            result = Parallel(n_jobs=10, prefer='threads')(
                delayed(self.parallel_model_performance_calculation)(polluted_feature_result.get_random_seed(), polluted_feature_result.get_pollution_level(), self.ml_algorithm, polluted_feature_result.get_train_df(), polluted_feature_result.get_test_df())
                for polluted_feature_result in polluted_feature_results)
            results_df = pd.concat([results_df, pd.concat(result)], ignore_index=True)
            self.experiment_results[feature] = results_df
        return self.experiment_results

    def parallel_model_performance_calculation(self, random_seed: int, pollution_level: float, experiment, train_df_polluted: DataFrame, test_df_polluted: DataFrame):
        np.random.seed(random_seed)

        exp = experiment(train_df_polluted, test_df_polluted, self.metadata[self.ds_name], 'multi_error', self.pre_pollution_setting_id)
        results = exp.run('', 'scenario', explain=False)

        new_results_row = pd.Series({'pollution_level': pollution_level,
                                     'real_f1': results[exp.name]['scoring']['macro avg']['f1-score'],
                                     'predicted_f1': 0.0,
                                     'random_seed': random_seed})

        return new_results_row.to_frame().T
