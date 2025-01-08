from typing import Dict, List
from classification.utils.artifical_pollution_new import PollutedFeature


class CleaningConfig:
    def __init__(self, regression_results, polluted_dfs):
        self.config = {}
        self._create_config(regression_results, polluted_dfs)

    def _create_config(self, regression_results: Dict, all_polluted_features_results: Dict[str, List[PollutedFeature]]):
        for feature in regression_results:
            cleaning_candidates_train = []
            cleaning_candidates_test = []
            polluted_feature_results = all_polluted_features_results[feature]
            for polluted_feature_result in polluted_feature_results:
                cleaning_candidates_train.extend(polluted_feature_result.get_sampling_indexes_train())
                cleaning_candidates_test.extend(polluted_feature_result.get_sampling_indexes_test())
            self.config[feature] = {
                'data': regression_results[feature],
                'cleaning_candidates_train': cleaning_candidates_train,
                'cleaning_candidates_test': cleaning_candidates_test
            }

    def get_configs(self):
        return self.config

    def get_config(self, feature):
        return self.config[feature]
