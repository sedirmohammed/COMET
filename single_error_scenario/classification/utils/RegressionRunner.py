from classification.utils.regression_model import *


class RegressionRunner:
    def __init__(self, pollution_results):
        self.pollution_results = pollution_results
        self.regression_results = {}

    def run(self):
        for feature in self.pollution_results:
            feature_pollution_results = self.pollution_results[feature]
            regression_model = RegressionModel(feature_pollution_results, {}, '')
            results_df = regression_model.fit_regression_models()
            self.regression_results[feature] = results_df
            del regression_model
            import gc
            gc.collect()
        return self.regression_results
