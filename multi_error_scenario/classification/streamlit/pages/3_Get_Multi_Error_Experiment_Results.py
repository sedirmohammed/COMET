import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from classification.utils.util import load_data_from_db
from classification.experiments import MultilayerPerceptronExperiment, SupportVectorMachineExperiment, GradientBoostingExperiment, KNeighborsExperiment

from config.definitions import ROOT_DIR
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


DS_NAME = ['SouthGermanCredit.csv', 'cmc.data', 'TelcoCustomerChurn.csv', 'EEG.arff']
DATABASE_URL = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
DATABASE_ENGINE = create_engine(DATABASE_URL, echo=True)
ML_ALGORITHMS = [KNeighborsExperiment.__name__, SupportVectorMachineExperiment.__name__, GradientBoostingExperiment.__name__, MultilayerPerceptronExperiment.__name__]
ML_ALGORITHMS_NAMES = ['SGDClassifierExperimentSVM', 'SGDClassifierExperimentLinearRegression', 'SGDClassifierExperimentLogisticRegression']
ML_ALGORITHMS_NAMES_SHORT = {KNeighborsExperiment.__name__: 'KNN', SupportVectorMachineExperiment.__name__: 'SVM',
                    GradientBoostingExperiment.__name__: 'GB', MultilayerPerceptronExperiment.__name__: 'MLP',
                    'SGDClassifierExperimentSVM': 'AC\nSVM', 'SGDClassifierExperimentLinearRegression': 'LIR', 'SGDClassifierExperimentLogisticRegression': 'LOR'}
error_type_names = {'CategoricalShiftModifier': 'Categorical Shift', 'GaussianNoiseModifier': 'Gaussian Noise',
                    'MissingValuesModifier': 'Missing Values', 'ScalingModifier': 'Scaling'}
error_type_names_short = {'CategoricalShiftModifier': 'CS', 'GaussianNoiseModifier': 'GN',
                    'MissingValuesModifier': 'MV', 'ScalingModifier': 'S'}
def load_and_filter_data(base_name, ds_name, experiment_str, error_type_str, database_url):
    data = load_data_from_db(f'{base_name}_{ds_name}_{experiment_str}', database_url)
    if len(data) != 0:
        if 'dataset' in data.columns:
            data = data[
                (data['dataset'] == ds_name) &
                (data['experiment'] == experiment_str)
            ]
    return data


st.set_page_config(
    page_title='COMET',
    page_icon='✔',
    layout='wide'
)


def get_comet_df(table_name_prefix='cleaning_results'):
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
                # mark current row as used_budget=-1
                comet_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            comet_df.at[previous_row_index, 'f1_score'] = row['f1_score']
    # delete rows with used_budget=-1
    comet_df = comet_df[comet_df['used_budget'] != -1]

    comet_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)

    comet_df['used_budget'] = comet_df['used_budget'].astype(int)
    comet_df['used_budget'] = comet_df.groupby(['pre_pollution_setting_id'])['used_budget'].cumsum()

    return comet_df


def get_rr_df():
    rr_df = load_and_filter_data('cleaning_schedule_completely_random', ds_name, ml_algorithm_str, '', DATABASE_URL)
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
                # mark current row as used_budget=-1
                rr_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            rr_df.at[previous_row_index, 'f1_score'] = row['f1_score']
    # delete rows with used_budget=-1
    rr_df = rr_df[rr_df['used_budget'] != -1]

    rr_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)

    rr_df['used_budget'] = rr_df['used_budget'].astype(int)
    rr_df['used_budget'] = rr_df.groupby(['run', 'pre_pollution_setting_id'])['used_budget'].cumsum()

    return rr_df


def get_fir_df():
    fir_df = load_and_filter_data('cleaning_schedule_static_features_importance', ds_name, ml_algorithm_str,'', DATABASE_URL)
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
                # mark current row as used_budget=-1
                fir_df.at[index, 'used_budget'] = -1
            else:
                print('Previous row not found')
            fir_df.at[previous_row_index, 'f1_score'] = row['f1_score']

    # delete rows with used_budget=-1
    fir_df = fir_df[fir_df['used_budget'] != -1]

    fir_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)

    fir_df['used_budget'] = fir_df['used_budget'].astype(int)
    fir_df['used_budget'] = fir_df.groupby(['pre_pollution_setting_id'])['used_budget'].cumsum()
    return fir_df


sns.set_context("paper")
sns.set(style="whitegrid", font_scale=2.1)


def get_cleaned_df(comet_df):
    cleaned_df = load_and_filter_data('cleaned_data_results', ds_name, ml_algorithm_str,'', DATABASE_URL)
    if len(cleaned_df) != 0:
        cleaned_df.rename(columns={'polluter': 'error_type'}, inplace=True)
        cleaned_df.rename(columns={'real_f1': 'f1_score'}, inplace=True)
        cleaned_df.drop(columns=['experiment', 'dataset'], inplace=True)
        # join cleaned_df and comet_df, over error_type, pre_pollution_setting_id
        cleaned_df = cleaned_df.merge(comet_df, on=['pre_pollution_setting_id'], how='left', suffixes=('_cleaned', '_comet'))
        cleaned_df = cleaned_df[cleaned_df['iteration'] == 0]
        # calculate diff between f1_score_cleaned and f1_score_comet
        cleaned_df['f1_score_diff'] = cleaned_df['f1_score_cleaned'] - cleaned_df['f1_score_comet']
        cleaned_df = cleaned_df.groupby(['error_type']).agg({'f1_score_diff': ['mean']}).reset_index()
        cleaned_df.columns = ['error_type', 'f1_score_diff_mean']
    return cleaned_df


all_diffs_df = pd.DataFrame()

for ds_name in DS_NAME:
    for ml_algorithm_str in ML_ALGORITHMS:
        #break
        st.header(f'ML Algorithm: {ml_algorithm_str}; Dataset: {ds_name}')

        comet_df = get_comet_df(table_name_prefix='cleaning_results')
        comet_light_df = get_comet_df(table_name_prefix='comet_light_cleaning_results')
        rr_df = get_rr_df()
        fir_df = get_fir_df()
        cleaned_df = get_cleaned_df(comet_df)

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

        # join diff_comet_rr and diff_comet_fir, over used_budget, error_type, pre_pollution_setting_id
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
        diffs_df.drop('ml_algorithm', axis=1, inplace=True)

        plot_df = diffs_df.copy()
        pivot_df = plot_df.pivot_table(index='used_budget', columns='f1_cat', values='value', aggfunc='sum', fill_value=0)

        sns.set(style="whitegrid")

        custom_palette = {
            'f1_score_diff_fir_mean': '#029E73',
            'f1_score_diff_rr_mean': '#DE8F05',
            'f1_score_diff_comet_light_mean': '#b20101'
        }
        # Plotting
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

        # Getting the existing legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Updating the legend with the new handle
        if ds_name == 'SouthGermanCredit.csv' or ds_name == 'Titanic':
            plt.legend(handles, ['FIR', 'RR', 'CL'], fontsize=35, loc='upper right').set_visible(True)
        else:
            plt.legend(handles, ['FIR', 'RR', 'CL'], fontsize=35).set_visible(False)
        plt.xlabel('Used Budget', fontsize=35)
        plt.ylabel('F1 Advantage', fontsize=35)
        plt.xticks(ticks=[0, 10, 20, 30, 40, 50], labels=['0', '10', '20', '30', '40', '50'], fontsize=35)
        plt.yticks(ticks=[-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.15], labels=['-0.02', '0.0', '0.02', '0.04', '0.06', '0.08', '0.15'], fontsize=35)
        plt.tight_layout()
        plt.savefig(f'{ROOT_DIR}/paper/figures/comet_comparison_{ds_name}_{ml_algorithm_str}_bl.pdf')
        #st.pyplot(plt)

def get_activeclean_df():
    ac_df = load_and_filter_data('activeclean_results', ds_name, ml_algorithm_str,'', DATABASE_URL)

    ac_df = ac_df[['f1_score', 'used_budget', 'iteration', 'pre_pollution_setting_id']]
    ac_df['used_budget'] = ac_df['used_budget'].fillna(0)
    st.write(ac_df)

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

    # delete rows with used_budget=-1
    ac_df = ac_df[ac_df['used_budget'] != -1]

    ac_df.sort_values(by=['pre_pollution_setting_id', 'iteration', 'number_of_cleaned_cells'], inplace=True)
    ac_df['used_budget'] = ac_df['used_budget'].astype(int)

    ac_df['used_budget'] = ac_df.groupby(['pre_pollution_setting_id'])['used_budget'].cumsum()
    ac_df = ac_df[ac_df['used_budget'] <= 50]
    st.write(ac_df)
    return ac_df


for ds_name in DS_NAME:
    for ml_algorithm_str in ML_ALGORITHMS_NAMES:
        st.header(f'ML Algorithm: {ml_algorithm_str}; Dataset: {ds_name}')

        comet_df = get_comet_df()
        ac_df = get_activeclean_df()
        cleaned_df = get_cleaned_df(comet_df)

        if  len(ac_df) == 0 or len(comet_df) == 0:
            st.write(f'No data found for {ml_algorithm_str} and {ds_name}')
            continue

        diff_comet_ac = ac_df.merge(comet_df, on=['used_budget', 'pre_pollution_setting_id'], how='left', suffixes=('_ac', '_comet'))
        st.write(diff_comet_ac)

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
        diffs_df.drop('ml_algorithm', axis=1, inplace=True)


        sns.set(style="whitegrid")
        sns.set_context("paper", font_scale=1)
        custom_palette = {
            'f1_score_diff_ac_mean': '#FFC107'
        }
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(data=diffs_df, x='used_budget', y='value', hue='f1_cat', estimator=sum, errorbar=None)

        for p in barplot.patches:
            bar_value = p.get_height()
            if bar_value < 0:
                p.set_color('#D55E00')  # Set the bar color to red if the value is negative
            else:
                p.set_color('#a1c9f4')

        # if len(cleaned_df) != 0:
        #     cleaned_df_error_type = cleaned_df[cleaned_df['error_type'] == error_type]
        #     for index, row in cleaned_df_error_type.iterrows():
        #         break
        #         plt.axhline(y=row['f1_score_diff_mean'], color='r', linestyle='--')

        #custom_line = [Line2D([0], [0], color='r', linestyle='--', label='Cleaned')]

        # Getting the existing legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()


        # Updating the legend with the new handle
        plt.legend(handles, ['AC', 'Cleaned x Dirty']).set_visible(False)

        plt.xlabel('Used Budget', fontsize=35)
        plt.ylabel('F1 Advantage', fontsize=35)
        plt.xticks(ticks=[0, 10, 20, 30, 40, 50], labels=['0', '10', '20', '30', '40', '50'], fontsize=35)
        plt.yticks(ticks=[-0.1, 0.0, 0.2, 0.4, 0.5], labels=['-0.1', '0.0', '0.2', '0.4', '0.5'], fontsize=35)
        plt.tight_layout()
        plt.savefig(f'{ROOT_DIR}/paper/figures/ac_comparison_{ds_name}_{ml_algorithm_str}_ac.pdf')
        #st.pyplot(plt)
