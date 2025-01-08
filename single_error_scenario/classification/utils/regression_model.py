from abc import ABC, abstractmethod

#import pytensor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pymc as pm
import numpy as np
#import jax
#jax.config.update('jax_default_device', jax.devices('cpu')[0])
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

class BaseRegressionModel(ABC):
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class LinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        self.model = LinearRegression()
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict([[x]])


class PolynomialRegressionModel(BaseRegressionModel):
    def __init__(self, degree):
        self.model = None
        self.degree = degree
        self.x_poly = None

    def fit(self, x, y):
        polynomial_features = PolynomialFeatures(degree=self.degree)
        self.x_poly = polynomial_features.fit_transform(x)
        self.model = LinearRegression()
        self.model.fit(self.x_poly, y)

    def predict(self, x):
        polynomial_features = PolynomialFeatures(degree=self.degree)
        x_poly = polynomial_features.fit_transform([[x]])
        return self.model.predict(x_poly), [0, 0]


class BaseBayesianRegressionModel(ABC):
    def __init__(self):
        self.trace = None
        self.model = None

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class BayesianPolynomialRegressionModel(BaseBayesianRegressionModel):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree)

    def fit(self, x, y):
        rng = np.random.default_rng(42)  # Updated seed setting
        x_poly = self.poly.fit_transform(x)
        unique_x = np.unique(x)

        # Calculate standard deviation for each unique x value
        std_devs = {xi: np.std(y[x == xi]) for xi in unique_x}

        # Replace each y value with its group's standard deviation
        sigma_obs = np.array([std_devs[xi[0]] for xi in x]).reshape(y.shape)
        # for each entry in y if the value in [] is 0.0 replace with 1e-6
        sigma_obs[sigma_obs == 0.0] = 1e-6
        # set weight ehere x is 1 to 3, 2 to 2 and 3 to 1
        weights = np.array([3] * 10 + [2] * 10 + [1] * 10)

        sigma_obs = np.sqrt((sigma_obs ** 4) + 0.01)
        with pm.Model() as self.model:
            alpha = pm.Normal('alpha', mu=y[0][0], sigma=0.1)
            betas = pm.Normal('betas', mu=1, sigma=10, shape=(self.degree + 1,))
            sigma = pm.HalfNormal('sigma', sigma=1, testval=1.)
            mu = alpha + pm.math.dot(betas, x_poly.T)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=y)
            self.trace = pm.sample(250,  tune=500, chains=5, cores=5, target_accept=0.9, return_inferencedata=True, random_seed=rng)

    def predict(self, x):
        x_poly = self.poly.transform(np.array(x).reshape(-1, 1))
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        beta_samples = self.trace.posterior['betas'].values.reshape(-1, self.degree + 1)
        y_preds = np.dot(beta_samples, x_poly.T) + alpha_samples[:, None]
        mean_prediction = np.mean(y_preds, axis=0)
        lower_bound, upper_bound = np.percentile(y_preds, [2.5, 97.5], axis=0)
        return mean_prediction[0], [lower_bound[0], upper_bound[0]]


def predict_test(x, y, degree):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    sigma_obs = np.sqrt((1e-10 ** 4) + 0.01)
    with pm.Model():
        alpha = pm.Normal('alpha', mu=y[0][0], sigma=0.1)
        betas = pm.Normal('betas', mu=1, sigma=10, shape=(2 + 1,))
        sigma = pm.HalfNormal('sigma', sigma=1, testval=1.)
        mu = alpha + pm.math.dot(betas, x_poly.T)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=y)
        trace = pm.sample(250, nuts_sampler="numpyro", tune=500, chains=5, target_accept=0.9,
                          return_inferencedata=False, random_seed=42)

        results_df = pd.DataFrame(columns=['pollution_level', 'real_f1', 'predicted_f1'])
        results_df = pd.concat([results_df, pd.DataFrame({'pollution_level': x.flatten(), 'real_f1': y.flatten()})],
                               ignore_index=True)

        #self.agg_results_df = self.agg_results_df.append({'pollution_level': -0.01}, ignore_index=True)
        results_df = pd.concat([results_df, pd.DataFrame({'pollution_level': [-0.01]})], ignore_index=True)
        for x_level in [-0.01, 0.0, 0.01, 0.02]:
            x_poly = poly.transform(np.array(x_level).reshape(-1, 1))
            alpha_samples = trace.posterior['alpha'].values.flatten()
            beta_samples = trace.posterior['betas'].values.reshape(-1, 2 + 1)
            y_preds = np.dot(beta_samples, x_poly.T) + alpha_samples[:, None]
            mean_prediction = np.mean(y_preds, axis=0)
            lower_bound, upper_bound = np.percentile(y_preds, [2.5, 97.5], axis=0)
            print(
                f"mean_prediction: {mean_prediction[0]}, lower_bound: {lower_bound[0]}, upper_bound: {upper_bound[0]}")
            results_df.loc[results_df.pollution_level == x_level, 'predicted_f1'] = mean_prediction[0]
            results_df.loc[results_df.pollution_level == x_level, 'lower_confidence_border'] = lower_bound[0]
            results_df.loc[results_df.pollution_level == x_level, 'upper_confidence_border'] = upper_bound[0]
        results_df = results_df.sort_values(by=['pollution_level']).reset_index(drop=True)
        results_df = results_df.groupby(['pollution_level']).mean().reset_index()
        return results_df


class RegressionModel:
    def __init__(self, agg_results_df, feature_wise_pollution_level, feature):
        self.agg_results_df = agg_results_df
        self.feature_wise_pollution_level = feature_wise_pollution_level
        self.feature = feature
        self.poly_reg_model = PolynomialRegressionModel(degree=2)
        self.bayesian_reg_model = BayesianPolynomialRegressionModel(degree=2)

    def fit_linear_regression(self, x, y):
        lin_reg_model = LinearRegressionModel()
        lin_reg_model.fit(x, y)
        return lin_reg_model

    def fit_polynomial_regression(self, x, y, degree=2):
        poly_reg_model = PolynomialRegressionModel(degree=degree)
        poly_reg_model.fit(x, y)
        return poly_reg_model

    def fit_bayesian_regression(self, x, y, degree=2):
        poly_reg_model = BayesianPolynomialRegressionModel(degree=degree)
        poly_reg_model.fit(x, y)
        return poly_reg_model

    def predict_value(self, regression_model, x):
        return regression_model.predict(x)#[0][0]

    def predict_value_with_polynomial_regression(self, x):
        return self.poly_reg_model.predict(x)[0][0]

    def fit_regression_models(self):
        filtered_rows = self.agg_results_df[self.agg_results_df['pollution_level'] >= 0.0].copy()
        filtered_rows.sort_values(by=['pollution_level'], inplace=True)
        x = filtered_rows[['pollution_level']]
        y = filtered_rows[['real_f1']]

        x['pollution_level_unknown'] = x['pollution_level'].rank(method='dense', ascending=True).astype(int)
        #poly_reg_model_deg_2 = self.fit_polynomial_regression(x['pollution_level_unknown'].values.reshape(-1, 1), y.values, degree=4)
        results_df = predict_test(x['pollution_level'].values.reshape(-1, 1), y.values, degree=2)
        return results_df
        bayesian_reg_model_deg_2 = self.fit_bayesian_regression(x['pollution_level'].values.reshape(-1, 1), y.values, degree=2)

        self.agg_results_df['lower_confidence_border'] = 0.0
        self.agg_results_df['upper_confidence_border'] = 0.0

        pollution_levels = self.agg_results_df['pollution_level'].unique()
        pollution_levels = np.append(pollution_levels, -0.01)

        self.agg_results_df = pd.concat([self.agg_results_df, pd.DataFrame({'pollution_level': [-0.01]})], ignore_index=True)
        series = pd.Series(pollution_levels)
        ranks = series.rank(method='dense').astype(int) - 1
        mapped_pollution_levels = ranks.tolist()
        for index, pollution_level in enumerate(pollution_levels):
            pollution_level = round(pollution_level, 2)

            #y_pred, conf_interval = self.predict_value(poly_reg_model_deg_2, mapped_pollution_levels[index])
            y_pred, conf_interval = self.predict_value(bayesian_reg_model_deg_2, pollution_level)
            self.agg_results_df['pollution_level'] = self.agg_results_df['pollution_level'].round(2)
            self.agg_results_df.loc[self.agg_results_df.pollution_level == pollution_level, 'predicted_f1'] = y_pred
            self.agg_results_df.loc[self.agg_results_df.pollution_level == pollution_level, 'lower_confidence_border'] = conf_interval[0]
            self.agg_results_df.loc[self.agg_results_df.pollution_level == pollution_level, 'upper_confidence_border'] = conf_interval[1]

        self.agg_results_df = self.agg_results_df.sort_values(by=['pollution_level']).reset_index(drop=True)
        self.agg_results_df = self.agg_results_df.groupby(['pollution_level']).mean().reset_index().drop('random_seed', axis=1)
        return self.agg_results_df