import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from classification.utils.util import load_data_from_db
from classification.experiments import DecisionTreeExperiment, \
    MultilayerPerceptronExperiment, SupportVectorMachineExperiment, GradientBoostingExperiment, KNeighborsExperiment
from classification.utils.DatasetModifier import MissingValuesModifier, GaussianNoiseModifier, ScalingModifier, \
    CategoricalShiftModifier
from config.definitions import ROOT_DIR
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


DS_NAME = ['SouthGermanCredit.csv', 'cmc.data', 'TelcoCustomerChurn.csv', 'EEG.arff', 'Airbnb', 'Credit', 'Titanic']
DATABASE_URL = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
DATABASE_ENGINE = create_engine(DATABASE_URL, echo=True)
ML_ALGORITHMS = [KNeighborsExperiment.__name__, SupportVectorMachineExperiment.__name__, GradientBoostingExperiment.__name__, MultilayerPerceptronExperiment.__name__]
ERROR_TYPES = [CategoricalShiftModifier.get_classname(), GaussianNoiseModifier.get_classname(), MissingValuesModifier.get_classname(), ScalingModifier.get_classname()]
ML_ALGORITHMS_NAMES = ['SGDClassifierExperimentSVM', 'SGDClassifierExperimentLinearRegression', 'SGDClassifierExperimentLogisticRegression']
ML_ALGORITHMS_NAMES_SHORT = {KNeighborsExperiment.__name__: 'KNN', SupportVectorMachineExperiment.__name__: 'SVM',
                    GradientBoostingExperiment.__name__: 'GB', MultilayerPerceptronExperiment.__name__: 'MLP',
                    'SGDClassifierExperimentSVM': 'AC\nSVM', 'SGDClassifierExperimentLinearRegression': 'LIR', 'SGDClassifierExperimentLogisticRegression': 'LOR'}
error_type_names = {'CategoricalShiftModifier': 'Categorical Shift', 'GaussianNoiseModifier': 'Gaussian Noise',
                    'MissingValuesModifier': 'Missing Values', 'ScalingModifier': 'Scaling'}
error_type_names_short = {'CategoricalShiftModifier': 'CS', 'GaussianNoiseModifier': 'GN',
                    'MissingValuesModifier': 'MV', 'ScalingModifier': 'S'}
def load_and_filter_data(base_name, ds_name, experiment_str, error_type, database_url):

    if error_type == '':
        data = load_data_from_db(f'{base_name}_{ds_name}_{experiment_str}', database_url)
        if len(data) != 0:
            if 'dataset' in data.columns:
                data = data[
                    (data['dataset'] == ds_name) &
                    (data['experiment'] == experiment_str)
                    ]
    else:
        data = load_data_from_db(f'{base_name}_{ds_name}_{experiment_str}_{error_type}', database_url)
        if len(data) != 0:
            if 'dataset' in data.columns:
                data = data[
                    (data['dataset'] == ds_name) &
                    (data['polluter'] == error_type) &
                    (data['experiment'] == experiment_str)
                ]
    return data


st.set_page_config(
    page_title='COMET',
    page_icon='✔',
    layout='wide'
)


def get_comet_multi_error_df(table_name_prefix='cleaning_results'):
    comet_df = load_and_filter_data(table_name_prefix, ds_name, ml_algorithm_str, '', DATABASE_URL)
    if len(comet_df) == 0:
        return pd.DataFrame()
    comet_df.drop(columns=['ml_algorithm', 'feature', 'predicted_f1_score'], inplace=True)
    comet_df['error_type'] = comet_df['error_type'].astype('str')
    for index, row in comet_df.iterrows():
        if row['used_budget'] != 0 and row['used_budget'] > 1:
            for i in range(1, int(row['used_budget'])):
                temp_row = row.copy()
                previous_row = comet_df[
                        (comet_df['iteration'] == row['iteration'] - 1) &
                        (comet_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
                    ]
                temp_row['used_budget'] = 1
                temp_row['number_of_cleaned_cells'] = 0
                temp_row['f1_score'] = previous_row['f1_score'].values[0]
                temp_df = pd.DataFrame(temp_row).T
                comet_df = pd.concat([comet_df, temp_df.copy()], ignore_index=True)
            comet_df.at[index, 'used_budget'] = 1
        if row['used_budget'] == 0 and row['iteration'] != 0:
            previous_row_index = comet_df[
                (comet_df['iteration'] == row['iteration'] - 1) &
                (comet_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
            ].index
            if len(previous_row_index) != 0:
                previous_row_index = previous_row_index[0]
                comet_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            comet_df.at[previous_row_index, 'f1_score'] = row['f1_score']
    comet_df = comet_df[comet_df['used_budget'] != -1]

    comet_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)

    comet_df['used_budget'] = comet_df['used_budget'].astype(int)
    comet_df['used_budget'] = comet_df.groupby(['pre_pollution_setting_id'])['used_budget'].cumsum()
    return comet_df


def get_rr_multi_error_df():
    rr_df = load_and_filter_data('cleaning_schedule_completely_random', ds_name, ml_algorithm_str, '', DATABASE_URL)
    if len(rr_df) == 0:
        return pd.DataFrame()
    rr_df = rr_df[['real_f1', 'used_budget', 'iteration', 'pre_pollution_setting_id', 'run']]

    rr_df.rename(columns={'real_f1': 'f1_score'}, inplace=True)
    rr_df = rr_df.sort_values(by=['iteration'], ascending=True)
    rr_df['used_budget'] = rr_df['used_budget'].astype(int)
    rr_df['run'] = rr_df['run'].astype(int)
    rr_df['iteration'] = rr_df['iteration'].astype(int)


    for index, row in rr_df.iterrows():
        if row['used_budget'] != 0 and row['used_budget'] > 1:
            for i in range(1, int(row['used_budget'])):
                temp_row = row.copy()
                previous_row = rr_df[
                        (rr_df['iteration'] == row['iteration'] - 1) &
                        (rr_df['run'] == row['run']) &
                        (rr_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
                    ]
                temp_row['used_budget'] = 1
                temp_row['number_of_cleaned_cells'] = 0
                temp_row['f1_score'] = previous_row['f1_score'].values[0]
                temp_df = pd.DataFrame(temp_row).T
                rr_df = pd.concat([rr_df, temp_df.copy()], ignore_index=True)
            rr_df.at[index, 'used_budget'] = 1
        if row['used_budget'] == 0 and row['iteration'] != 0:
            previous_row_index = rr_df[
                (rr_df['iteration'] == row['iteration'] - 1) &
                (rr_df['run'] == row['run']) &
                (rr_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
            ].index
            if len(previous_row_index) != 0:
                previous_row_index = previous_row_index[0]
                rr_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            rr_df.at[previous_row_index, 'f1_score'] = row['f1_score']
    rr_df = rr_df[rr_df['used_budget'] != -1]

    rr_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)

    rr_df['used_budget'] = rr_df['used_budget'].astype(int)
    rr_df['used_budget'] = rr_df.groupby(['run', 'pre_pollution_setting_id'])['used_budget'].cumsum()
    return rr_df


def get_fir_multi_error_df():
    fir_df = load_and_filter_data('cleaning_schedule_static_features_importance', ds_name, ml_algorithm_str,'', DATABASE_URL)
    if len(fir_df) == 0:
        return pd.DataFrame()
    fir_df.rename(columns={'real_f1': 'f1_score'}, inplace=True)
    fir_df = fir_df[['f1_score', 'used_budget', 'iteration', 'error_type', 'pre_pollution_setting_id']]
    fir_df['used_budget'] = fir_df['used_budget'].fillna(0)


    for index, row in fir_df.iterrows():
        if row['used_budget'] != 0 and row['used_budget'] > 1:
            for i in range(1, int(row['used_budget'])):
                temp_row = row.copy()
                previous_row = fir_df[
                        (fir_df['iteration'] == row['iteration'] - 1) &
                        (fir_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
                    ]
                temp_row['used_budget'] = 1
                temp_row['number_of_cleaned_cells'] = 0
                temp_row['f1_score'] = previous_row['f1_score'].values[0]
                temp_df = pd.DataFrame(temp_row).T
                fir_df = pd.concat([fir_df, temp_df.copy()], ignore_index=True)
            fir_df.at[index, 'used_budget'] = 1
        if row['used_budget'] == 0 and row['iteration'] != 0:
            previous_row_index = fir_df[
                (fir_df['iteration'] == row['iteration'] - 1) &
                (fir_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
            ].index
            if len(previous_row_index) != 0:
                previous_row_index = previous_row_index[0]
                fir_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            fir_df.at[previous_row_index, 'f1_score'] = row['f1_score']

    fir_df = fir_df[fir_df['used_budget'] != -1]

    fir_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)

    fir_df['used_budget'] = fir_df['used_budget'].astype(int)
    fir_df['used_budget'] = fir_df.groupby(['pre_pollution_setting_id'])['used_budget'].cumsum()
    return fir_df


def get_activeclean_multi_error_df():
    ac_df = load_and_filter_data('activeclean_results', ds_name, ml_algorithm_str,'', DATABASE_URL)

    if len(ac_df) == 0:
        return pd.DataFrame()

    ac_df = ac_df[['f1_score', 'used_budget', 'iteration', 'pre_pollution_setting_id']]
    ac_df['used_budget'] = ac_df['used_budget'].fillna(0)

    for index, row in ac_df.iterrows():
        if row['used_budget'] != 0 and row['used_budget'] > 1:
            for i in range(1, int(row['used_budget'])):
                temp_row = row.copy()
                previous_row = ac_df[
                        (ac_df['iteration'] == row['iteration'] - 1) &
                        (ac_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
                    ]
                temp_row['used_budget'] = 1
                temp_row['number_of_cleaned_cells'] = 0
                temp_row['f1_score'] = previous_row['f1_score'].values[0]
                temp_df = pd.DataFrame(temp_row).T
                ac_df = pd.concat([ac_df, temp_df.copy()], ignore_index=True)
            ac_df.at[index, 'used_budget'] = 1
        if row['used_budget'] == 0 and row['iteration'] != 0:
            previous_row_index = ac_df[
                (ac_df['iteration'] == row['iteration'] - 1) &
                (ac_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
            ].index
            if len(previous_row_index) != 0:
                previous_row_index = previous_row_index[0]
                # mark current row as used_budget=-1
                ac_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            ac_df.at[previous_row_index, 'f1_score'] = row['f1_score']

    ac_df = ac_df[ac_df['used_budget'] != -1]
    ac_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)
    ac_df['used_budget'] = ac_df['used_budget'].astype(int)

    ac_df['used_budget'] = ac_df.groupby(['pre_pollution_setting_id'])['used_budget'].cumsum()
    ac_df = ac_df[ac_df['used_budget'] <= 50]
    st.write(ac_df)
    return ac_df


def get_comet_df(table_name_prefix='cleaning_results'):
    comet_df = pd.DataFrame()
    for error_type_name in ERROR_TYPES:
        df = load_and_filter_data(table_name_prefix, ds_name, ml_algorithm_str, error_type_name, DATABASE_URL)
        comet_df = pd.concat([comet_df, df], ignore_index=True)
    comet_df.drop(columns=['ml_algorithm', 'feature', 'predicted_f1_score'], inplace=True)
    comet_df['error_type'] = comet_df['error_type'].astype('str')
    for index, row in comet_df.iterrows():
        if row['used_budget'] != 0 and row['used_budget'] > 1:
            for i in range(1, int(row['used_budget'])):
                temp_row = row.copy()
                previous_row = comet_df[
                        (comet_df['iteration'] == row['iteration'] - 1) &
                        (comet_df['error_type'] == row['error_type']) &
                        (comet_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
                    ]
                temp_row['used_budget'] = 1
                temp_row['number_of_cleaned_cells'] = 0
                temp_row['f1_score'] = previous_row['f1_score'].values[0]
                temp_df = pd.DataFrame(temp_row).T
                comet_df = pd.concat([comet_df, temp_df.copy()], ignore_index=True)
            comet_df.at[index, 'used_budget'] = 1
        if row['used_budget'] == 0 and row['iteration'] != 0:
            previous_row_index = comet_df[
                (comet_df['iteration'] == row['iteration'] - 1) &
                (comet_df['error_type'] == row['error_type']) &
                (comet_df['pre_pollution_setting_id'] == row['pre_pollution_setting_id'])
            ].index
            if len(previous_row_index) != 0:
                previous_row_index = previous_row_index[0]
                comet_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            comet_df.at[previous_row_index, 'f1_score'] = row['f1_score']
    comet_df = comet_df[comet_df['used_budget'] != -1]

    comet_df.sort_values(by=['error_type', 'pre_pollution_setting_id', 'iteration'], inplace=True)

    comet_df['used_budget'] = comet_df['used_budget'].astype(int)
    comet_df['used_budget'] = comet_df.groupby(['error_type', 'pre_pollution_setting_id'])['used_budget'].cumsum()

    return comet_df


def get_rr_df(comet_df):
    rr_df = pd.DataFrame()
    for error_type_name in ERROR_TYPES:
        df = load_and_filter_data('cleaning_schedule_completely_random', ds_name, ml_algorithm_str, error_type_name,
                                  DATABASE_URL)
        rr_df = pd.concat([rr_df, df], ignore_index=True)
    rr_df = rr_df[['real_f1', 'used_budget', 'iteration', 'polluter', 'pre_pollution_setting_id']]
    rr_df.rename(columns={'real_f1': 'f1_score'}, inplace=True)
    rr_df.loc[rr_df['iteration'] == 0, 'used_budget'] = 0
    rr_df = rr_df.sort_values(by=['iteration'], ascending=True)
    rr_df['used_budget'] = rr_df['used_budget'].astype(int)
    rr_df = rr_df.groupby(['iteration', 'polluter', 'pre_pollution_setting_id']).head(5).reset_index(drop=True)
    rr_df = rr_df[rr_df['iteration'] <= 50]
    rr_df = rr_df.groupby(['iteration', 'polluter', 'pre_pollution_setting_id']).agg({'f1_score': ['mean', 'std'], 'used_budget': 'mean'}).reset_index()
    rr_df.columns = ['iteration', 'error_type', 'pre_pollution_setting_id', 'f1_score', 'f1_score_std', 'used_budget']
    rr_df.sort_values(by=['error_type', 'pre_pollution_setting_id', 'iteration'], inplace=True)
    rr_df.drop('f1_score_std', axis=1, inplace=True)
    rr_df['used_budget'] = rr_df['used_budget'].astype(int)
    rr_df['used_budget'] = rr_df.groupby(['error_type', 'pre_pollution_setting_id'])[
        'used_budget'].cumsum()
    for error_type_name in ERROR_TYPES:
        for pre_pollution_setting_id in rr_df[rr_df['error_type'] == error_type_name]['pre_pollution_setting_id'].unique():
            if len(comet_df[(comet_df['error_type'] == error_type_name) & (comet_df['pre_pollution_setting_id'] == pre_pollution_setting_id)]) == 0:
                rr_df = rr_df[
                    ~((rr_df['error_type'] == error_type_name) & (rr_df['pre_pollution_setting_id'] == pre_pollution_setting_id))
                ]
                continue

            original_f1_score = comet_df[
                (comet_df['error_type'] == error_type_name) &
                (comet_df['pre_pollution_setting_id'] == pre_pollution_setting_id) &
                (comet_df['iteration'] == 0)
            ]['f1_score'].values[0]
            temp_df = pd.DataFrame({'iteration': 0, 'error_type': error_type_name, 'pre_pollution_setting_id': pre_pollution_setting_id, 'f1_score': original_f1_score, 'used_budget': 0}, index=[0])
            rr_df = pd.concat([rr_df, temp_df], ignore_index=True)
    rr_df.sort_values(by=['error_type', 'pre_pollution_setting_id', 'iteration'], inplace=True)
    return rr_df


def get_fir_df():
    fir_df = pd.DataFrame()
    for error_type_name in ERROR_TYPES:
        df = load_and_filter_data('cleaning_schedule_static_features_importance', ds_name, ml_algorithm_str,
                                  error_type_name,
                                  DATABASE_URL)
        fir_df = pd.concat([fir_df, df], ignore_index=True)
    fir_df = fir_df[['f1_score', 'used_budget', 'iteration', 'polluter', 'pre_pollution_setting_id']]
    fir_df.rename(columns={'polluter': 'error_type'}, inplace=True)
    fir_df['used_budget'] = fir_df['used_budget'].fillna(0)
    fir_df['iteration'] = fir_df['iteration'].astype(int)
    fir_df['used_budget'] = fir_df['used_budget'].astype(float)
    fir_df['f1_score'] = fir_df['f1_score'].astype(float)
    fir_df = fir_df[fir_df['iteration'] <= 50]
    fir_df.sort_values(by=['error_type', 'pre_pollution_setting_id', 'iteration'], inplace=True)
    fir_df['used_budget'] = fir_df.groupby(['error_type', 'pre_pollution_setting_id'])['used_budget'].cumsum()
    return fir_df

sns.set_context("paper")
sns.set(style="whitegrid", font_scale=2.1)


def get_cleaned_df(comet_df):
    cleaned_df = pd.DataFrame()
    for error_type_name in ERROR_TYPES:
        df = load_and_filter_data('cleaned_data_results', ds_name, ml_algorithm_str,
                                  error_type_name,
                                  DATABASE_URL)
        cleaned_df = pd.concat([cleaned_df, df], ignore_index=True)
    if len(cleaned_df) != 0:
        cleaned_df.rename(columns={'polluter': 'error_type'}, inplace=True)
        cleaned_df.rename(columns={'real_f1': 'f1_score'}, inplace=True)
        cleaned_df.drop(columns=['experiment', 'dataset'], inplace=True)
        cleaned_df = cleaned_df.merge(comet_df, on=['error_type', 'pre_pollution_setting_id'], how='left', suffixes=('_cleaned', '_comet'))
        cleaned_df = cleaned_df[cleaned_df['iteration'] == 0]
        cleaned_df['f1_score_diff'] = cleaned_df['f1_score_cleaned'] - cleaned_df['f1_score_comet']
        cleaned_df = cleaned_df.groupby(['error_type']).agg({'f1_score_diff': ['mean']}).reset_index()
        cleaned_df.columns = ['error_type', 'f1_score_diff_mean']
    return cleaned_df



all_diffs_df = pd.DataFrame()

for ds_name in DS_NAME:
    for ml_algorithm_str in ML_ALGORITHMS:
        st.header(f'ML Algorithm: {ml_algorithm_str}; Dataset: {ds_name}')

        comet_df = get_comet_df(table_name_prefix='cleaning_results')
        comet_light_df = get_comet_df(table_name_prefix='comet_light_cleaning_results')
        rr_df = get_rr_df(comet_df)
        fir_df = get_fir_df()
        cleaned_df = get_cleaned_df(comet_df)

        diff_comet_rr = rr_df.merge(comet_df, on=['used_budget', 'error_type', 'pre_pollution_setting_id'], how='left', suffixes=('_rr', '_comet'))
        diff_comet_rr = diff_comet_rr[['error_type', 'pre_pollution_setting_id', 'f1_score_rr', 'f1_score_comet', 'used_budget']]
        diff_comet_rr['f1_score_diff'] = diff_comet_rr['f1_score_comet'] - diff_comet_rr['f1_score_rr']

        diff_comet_fir = fir_df.merge(comet_df, on=['used_budget', 'error_type', 'pre_pollution_setting_id'], how='left', suffixes=('_fir', '_comet'))
        diff_comet_fir = diff_comet_fir[['error_type', 'pre_pollution_setting_id', 'f1_score_fir', 'f1_score_comet', 'used_budget']]
        diff_comet_fir['f1_score_diff'] = diff_comet_fir['f1_score_comet'] - diff_comet_fir['f1_score_fir']

        diff_comet_comet_light = comet_light_df.merge(comet_df, on=['used_budget', 'error_type', 'pre_pollution_setting_id'], how='left', suffixes=('_comet_light', '_comet'))
        diff_comet_comet_light = diff_comet_comet_light[['error_type', 'pre_pollution_setting_id', 'f1_score_comet_light', 'f1_score_comet', 'used_budget']]
        diff_comet_comet_light['f1_score_diff'] = diff_comet_comet_light['f1_score_comet'] - diff_comet_comet_light['f1_score_comet_light']
        diff_comet_comet_light.rename(columns={'f1_score_diff': 'f1_score_diff_comet_light'}, inplace=True)
        diff_comet_comet_light = diff_comet_comet_light[['error_type', 'pre_pollution_setting_id', 'f1_score_diff_comet_light', 'used_budget']]

        diffs_df = diff_comet_rr.merge(diff_comet_fir, on=['used_budget', 'error_type', 'pre_pollution_setting_id'], how='left', suffixes=('_rr', '_fir'))
        diffs_df = diffs_df.merge(diff_comet_comet_light, on=['error_type', 'used_budget', 'pre_pollution_setting_id'], how='left')
        diffs_df = diffs_df[['error_type', 'pre_pollution_setting_id', 'f1_score_diff_fir', 'f1_score_diff_rr', 'f1_score_diff_comet_light', 'used_budget']]
        diffs_df = diffs_df.groupby(['used_budget', 'error_type']).agg({'f1_score_diff_fir': ['mean', 'std'], 'f1_score_diff_rr': ['mean', 'std'], 'f1_score_diff_comet_light': ['mean', 'std']}).reset_index()
        diffs_df.columns = ['used_budget', 'error_type', 'f1_score_diff_fir_mean', 'f1_score_diff_fir_std', 'f1_score_diff_rr_mean', 'f1_score_diff_rr_std', 'f1_score_diff_comet_light_mean', 'f1_score_diff_comet_light_std']

        error_types = diffs_df['error_type'].unique()
        diffs_df.sort_values(by=['used_budget', 'error_type'], inplace=True)

        data_df = diffs_df.copy()
        diffs_df = diffs_df.drop(['f1_score_diff_fir_std', 'f1_score_diff_rr_std', 'f1_score_diff_comet_light_std'], axis=1)
        diffs_df = diffs_df.melt(id_vars=['used_budget', 'error_type'], var_name='f1_cat', value_name='value')

        diffs_df.sort_values(by=['used_budget', 'error_type'], inplace=True)

        diffs_df['ml_algorithm'] = ml_algorithm_str
        all_diffs_df = pd.concat([all_diffs_df, diffs_df], ignore_index=True)
        diffs_df.drop('ml_algorithm', axis=1, inplace=True)

        for error_type in diffs_df.error_type.unique():
            plot_df = diffs_df[diffs_df['error_type'] == error_type]
            pivot_df = plot_df.pivot_table(index='used_budget', columns='f1_cat', values='value', aggfunc='sum', fill_value=0)

            sns.set(style="whitegrid")

            custom_palette = {
                'f1_score_diff_fir_mean': '#029E73',
                'f1_score_diff_rr_mean': '#DE8F05',
                'f1_score_diff_comet_light_mean': '#b20101'
            }
            plt.figure(figsize=(10, 6))
            barplot = sns.barplot(data=plot_df, x='used_budget', y='value', hue='f1_cat', estimator=sum, errorbar=None,
                                  palette=custom_palette, width=0.99)
            if len(cleaned_df) != 0 and False == True:
                cleaned_df_error_type = cleaned_df[cleaned_df['error_type'] == error_type]
                for index, row in cleaned_df_error_type.iterrows():
                    continue
                    plt.axhline(y=row['f1_score_diff_mean'], color='r', linestyle='--')
            for patch in barplot.patches:
                patch.set_linewidth(0)

            handles, labels = plt.gca().get_legend_handles_labels()

            # Updating the legend with the new handle
            if ds_name == 'SouthGermanCredit.csv' and error_type == 'ScalingModifier' or ds_name == 'Titanic' and error_type == 'MissingValuesModifier':
                plt.legend(handles, ['FIR', 'RR', 'CL'], fontsize=35, loc='upper right').set_visible(True)
            else:
                plt.legend(handles, ['FIR', 'RR', 'CL'], fontsize=35).set_visible(False)
            #plt.title(f'F1 Score Differences for {error_type_names[error_type]}')
            plt.xlabel('Used Budget', fontsize=35)
            plt.ylabel('F1 Advantage', fontsize=35)
            plt.xticks(ticks=[0, 10, 20, 30, 40, 50], labels=['0', '10', '20', '30', '40', '50'], fontsize=35)
            plt.yticks(ticks=[-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.15], labels=['-0.02', '0.0', '0.02', '0.04', '0.06', '0.08', '0.15'], fontsize=35)
            plt.tight_layout()
            plt.savefig(f'{ROOT_DIR}/paper/figures/comet_comparison_{ds_name}_{ml_algorithm_str}_{error_type}_bl.pdf')
            #st.pyplot(plt)

def get_activeclean_df():
    ac_df = pd.DataFrame()
    for error_type_name in ERROR_TYPES:
        df = load_and_filter_data('activeclean_results', ds_name, ml_algorithm_str,
                                  error_type_name,
                                  DATABASE_URL)
        df['error_type'] = error_type_name
        ac_df = pd.concat([ac_df, df], ignore_index=True)
    return ac_df


for ds_name in DS_NAME:
    for ml_algorithm_str in ML_ALGORITHMS_NAMES:
        st.header(f'ML Algorithm: {ml_algorithm_str}; Dataset: {ds_name}')

        comet_df = get_comet_df()
        ac_df = get_activeclean_df()
        cleaned_df = get_cleaned_df(comet_df)

        diff_comet_ac = ac_df.merge(comet_df, on=['used_budget', 'error_type', 'pre_pollution_setting_id'], how='left', suffixes=('_ac', '_comet'))
        diff_comet_ac = diff_comet_ac[['error_type', 'pre_pollution_setting_id', 'f1_score_ac', 'f1_score_comet', 'used_budget']]
        diff_comet_ac['f1_score_diff_ac'] = diff_comet_ac['f1_score_comet'] - diff_comet_ac['f1_score_ac']

        diffs_df = diff_comet_ac.copy()
        diffs_df = diffs_df.groupby(['used_budget', 'error_type']).agg({'f1_score_diff_ac': ['mean', 'std']}).reset_index()
        diffs_df.columns = ['used_budget', 'error_type', 'f1_score_diff_ac_mean', 'f1_score_diff_ac_std']

        error_types = diffs_df['error_type'].unique()
        diffs_df['used_budget'] = diffs_df['used_budget'].astype(int)
        diffs_df.sort_values(by=['used_budget', 'error_type'], inplace=True)
        diffs_df = diffs_df.drop(['f1_score_diff_ac_std'], axis=1)
        diffs_df = diffs_df.melt(id_vars=['used_budget', 'error_type'], var_name='f1_cat', value_name='value')
        diffs_df.sort_values(by=['used_budget', 'error_type'], inplace=True)


        diffs_df['ml_algorithm'] = ml_algorithm_str
        all_diffs_df = pd.concat([all_diffs_df, diffs_df], ignore_index=True)
        diffs_df.drop('ml_algorithm', axis=1, inplace=True)

        for error_type in diffs_df.error_type.unique():
            plot_df = diffs_df[diffs_df['error_type'] == error_type]

            sns.set(style="whitegrid")
            sns.set_context("paper", font_scale=1)
            custom_palette = {
                'f1_score_diff_ac_mean': '#FFC107'
            }
            plt.figure(figsize=(10, 6))
            barplot = sns.barplot(data=plot_df, x='used_budget', y='value', hue='f1_cat', estimator=sum, errorbar=None)

            for p in barplot.patches:
                bar_value = p.get_height()
                if bar_value < 0:
                    p.set_color('#D55E00')  # Set the bar color to red if the value is negative
                else:
                    p.set_color('#a1c9f4')

            if len(cleaned_df) != 0:
                cleaned_df_error_type = cleaned_df[cleaned_df['error_type'] == error_type]
                for index, row in cleaned_df_error_type.iterrows():
                    break
                    plt.axhline(y=row['f1_score_diff_mean'], color='r', linestyle='--')
            from matplotlib.lines import Line2D

            custom_line = [Line2D([0], [0], color='r', linestyle='--', label='Cleaned')]

            handles, labels = plt.gca().get_legend_handles_labels()
            handles.extend(custom_line)

            plt.legend(handles, ['AC', 'Cleaned x Dirty']).set_visible(False)

            plt.xlabel('Used Budget', fontsize=35)
            plt.ylabel('F1 Advantage', fontsize=35)
            plt.xticks(ticks=[0, 10, 20, 30, 40, 50], labels=['0', '10', '20', '30', '40', '50'], fontsize=35)
            plt.yticks(ticks=[-0.1, 0.0, 0.2, 0.4, 0.5], labels=['-0.1', '0.0', '0.2', '0.4', '0.5'], fontsize=35)
            plt.tight_layout()
            plt.savefig(f'{ROOT_DIR}/paper/figures/ac_comparison_{ds_name}_{ml_algorithm_str}_{error_type}_ac.pdf')
            #st.pyplot(plt)


all_diffs_df = all_diffs_df[all_diffs_df['used_budget'] != 0]
print('Overall average', all_diffs_df.loc[:, 'value'].mean())

print(f"Max value: {all_diffs_df['value'].max()}")
print(f"Min value: {all_diffs_df['value'].min()}")

results_df = all_diffs_df.groupby(['error_type']).agg({'value': ['mean', 'std']}).reset_index()
results_df = results_df.drop(('value', 'std'), axis=1)
results_df.columns = results_df.columns.droplevel(1)
results_df['value'] = pd.to_numeric(results_df['value'], errors='coerce').round(2)
results_df['value'] = results_df['value'].round(2)
results_df['error_type'] = results_df['error_type'].map(error_type_names_short)
results_df['error_type'] = results_df['error_type'].str.replace(' ', '\n')
print(results_df.to_string())
results_df_1 = results_df.copy()



for ds_name in DS_NAME:
    for ml_algorithm_str in ML_ALGORITHMS_NAMES:
        st.header(f'ML Algorithm: {ml_algorithm_str}; Dataset: {ds_name}')

        comet_df = get_comet_multi_error_df()
        ac_df = get_activeclean_multi_error_df()

        if  len(ac_df) == 0 or len(comet_df) == 0:
            st.write(f'No data found for {ml_algorithm_str} and {ds_name}')
            continue

        diff_comet_ac = ac_df.merge(comet_df, on=['used_budget', 'pre_pollution_setting_id'], how='left', suffixes=('_ac', '_comet'))
        diff_comet_ac = diff_comet_ac[['pre_pollution_setting_id', 'f1_score_ac', 'f1_score_comet', 'used_budget']]
        diff_comet_ac['f1_score_diff_ac'] = diff_comet_ac['f1_score_comet'] - diff_comet_ac['f1_score_ac']

        diffs_df = diff_comet_ac.copy()
        diffs_df = diffs_df.groupby(['used_budget']).agg({'f1_score_diff_ac': ['mean', 'std']}).reset_index()
        diffs_df.columns = ['used_budget', 'f1_score_diff_ac_mean', 'f1_score_diff_ac_std']

        diffs_df['used_budget'] = diffs_df['used_budget'].astype(int)
        diffs_df.sort_values(by=['used_budget'], inplace=True)
        diffs_df = diffs_df.drop(['f1_score_diff_ac_std'], axis=1)
        diffs_df = diffs_df.melt(id_vars=['used_budget'], var_name='f1_cat', value_name='value')
        diffs_df.sort_values(by=['used_budget'], inplace=True)

        diffs_df['ml_algorithm'] = ml_algorithm_str
        all_diffs_df = pd.concat([all_diffs_df, diffs_df], ignore_index=True)


for ds_name in DS_NAME:
    for ml_algorithm_str in ML_ALGORITHMS:
        st.header(f'ML Algorithm: {ml_algorithm_str}; Dataset: {ds_name}')

        comet_df = get_comet_multi_error_df(table_name_prefix='cleaning_results')
        comet_light_df = get_comet_multi_error_df(table_name_prefix='comet_light_cleaning_results')
        rr_df = get_rr_multi_error_df()
        fir_df = get_fir_multi_error_df()

        if comet_df.empty or comet_light_df.empty or rr_df.empty or fir_df.empty:
            continue

        diff_comet_rr = rr_df.merge(comet_df, on=['used_budget', 'pre_pollution_setting_id'], how='left', suffixes=('_rr', '_comet'))
        diff_comet_rr = diff_comet_rr[['pre_pollution_setting_id', 'f1_score_rr', 'f1_score_comet', 'used_budget']]
        diff_comet_rr['f1_score_diff'] = diff_comet_rr['f1_score_comet'] - diff_comet_rr['f1_score_rr']

        diff_comet_fir = fir_df.merge(comet_df, on=['used_budget', 'pre_pollution_setting_id'], how='left', suffixes=('_fir', '_comet'))
        diff_comet_fir = diff_comet_fir[['pre_pollution_setting_id', 'f1_score_fir', 'f1_score_comet', 'used_budget']]
        diff_comet_fir['f1_score_diff'] = diff_comet_fir['f1_score_comet'] - diff_comet_fir['f1_score_fir']

        diff_comet_comet_light = comet_light_df.merge(comet_df, on=['used_budget', 'pre_pollution_setting_id'], how='left', suffixes=('_comet_light', '_comet'))
        diff_comet_comet_light = diff_comet_comet_light[['pre_pollution_setting_id', 'f1_score_comet_light', 'f1_score_comet', 'used_budget']]
        diff_comet_comet_light['f1_score_diff'] = diff_comet_comet_light['f1_score_comet'] - diff_comet_comet_light['f1_score_comet_light']
        diff_comet_comet_light.rename(columns={'f1_score_diff': 'f1_score_diff_comet_light'}, inplace=True)
        diff_comet_comet_light = diff_comet_comet_light[['pre_pollution_setting_id', 'f1_score_diff_comet_light', 'used_budget']]

        diffs_df = diff_comet_rr.merge(diff_comet_fir, on=['used_budget', 'pre_pollution_setting_id'], how='left', suffixes=('_rr', '_fir'))
        diffs_df = diffs_df.merge(diff_comet_comet_light, on=['used_budget', 'pre_pollution_setting_id'], how='left')
        diffs_df = diffs_df[['pre_pollution_setting_id', 'f1_score_diff_fir', 'f1_score_diff_rr', 'f1_score_diff_comet_light', 'used_budget']]
        diffs_df = diffs_df.groupby(['used_budget']).agg({'f1_score_diff_fir': ['mean', 'std'], 'f1_score_diff_rr': ['mean', 'std'], 'f1_score_diff_comet_light': ['mean', 'std']}).reset_index()
        diffs_df.columns = ['used_budget', 'f1_score_diff_fir_mean', 'f1_score_diff_fir_std', 'f1_score_diff_rr_mean', 'f1_score_diff_rr_std', 'f1_score_diff_comet_light_mean', 'f1_score_diff_comet_light_std']

        diffs_df.sort_values(by=['used_budget'], inplace=True)

        data_df = diffs_df.copy()

        diffs_df = diffs_df.drop(['f1_score_diff_fir_std', 'f1_score_diff_rr_std', 'f1_score_diff_comet_light_std'], axis=1)
        diffs_df = diffs_df.melt(id_vars=['used_budget'], var_name='f1_cat', value_name='value')

        diffs_df.sort_values(by=['used_budget'], inplace=True)

        diffs_df['ml_algorithm'] = ml_algorithm_str
        all_diffs_df = pd.concat([all_diffs_df, diffs_df], ignore_index=True)



results_df = all_diffs_df.groupby(['ml_algorithm']).agg({'value': ['mean', 'std']}).reset_index()
results_df = results_df.drop(('value', 'std'), axis=1)
results_df.columns = results_df.columns.droplevel(1)
results_df['value'] = pd.to_numeric(results_df['value'], errors='coerce').round(2)
results_df['value'] = results_df['value'].round(2)
results_df['ml_algorithm'] = results_df['ml_algorithm'].map(ML_ALGORITHMS_NAMES_SHORT)
all_diffs_df = all_diffs_df[all_diffs_df['used_budget'] != 0]
print('Overall average', all_diffs_df.loc[:, 'value'].mean())

print(f"Max value: {all_diffs_df['value'].max()}")
print(f"Min value: {all_diffs_df['value'].min()}")
print(results_df.to_string())
results_df_2 = results_df.copy()

preferred_order = ['GB', 'KNN', 'MLP', 'SVM', 'AC\nSVM', 'LIR', 'LOR']

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), sharex=False, sharey=True)

# Plotting the second bar plot
sns.barplot(x='ml_algorithm', y='value', data=results_df_2, color='#a1c9f4', ax=axs[0], order=preferred_order)
axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=30)
axs[0].set_title('(a) Grouped by ML algorithm', fontsize=35, fontweight='bold', pad=25)
axs[0].set_xlabel('', fontsize=1)
axs[0].set_ylabel('F1 Advantage', fontsize=35)
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=35)
axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


svm_index = preferred_order.index('SVM')
acs_index = preferred_order.index('AC\nSVM')
line_position = (svm_index + acs_index) / 2
axs[0].axvline(x=line_position, color='grey', linestyle='--', linewidth=2)

# Plotting the first bar plot
sns.barplot(x='error_type', y='value', data=results_df_1, color='#a1c9f4', ax=axs[1])
axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=35)
axs[1].set_title('(b) Grouped by Error Type', fontsize=35, fontweight='bold', pad=25)
axs[1].set_xlabel('', fontsize=1)
axs[1].set_ylabel('', fontsize=1)


plt.subplots_adjust(bottom=0.3)
plt.tight_layout()

plt.savefig(f'{ROOT_DIR}/paper/figures/combined_comparison.pdf')

print(results_df_1)
print(results_df_2)
