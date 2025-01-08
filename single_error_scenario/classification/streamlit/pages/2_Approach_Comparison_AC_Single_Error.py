import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from json import load as load_json
from classification.utils.util import load_data_from_db
from classification.experiments import SGDClassifierExperimentSVM, SGDClassifierExperimentLinearRegression, \
    SGDClassifierExperimentLogisticRegression
from classification.utils.DatasetModifier import MissingValuesModifier, GaussianNoiseModifier, ScalingModifier, \
    CategoricalShiftModifier
import plotly.graph_objects as go
from config.definitions import ROOT_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def plot_performance_per_budget_with_optimal(pre_pollution_setting_id, df, df_ac, df_cleaned):
    fig = go.Figure()

    if len(df) > 0:
        df = df.sort_values(by=['iteration'], ascending=True)
        df['used_budget'] = df['used_budget'].astype(int)
        df['used_budget'] = df['used_budget'].cumsum()
        df = df[df['used_budget'] <= 50]
        df = df.sort_values(by=['used_budget'], ascending=True)
        df = df.sort_values(by=['iteration', 'used_budget'], ascending=True)
        fig.add_trace(go.Scatter(x=df['used_budget'], y=df['f1_score'],
                                 mode='lines+markers',
                                 marker_color='#0173B2',
                                 name='COMET',
                                 line_shape='spline',
                                 text=df['feature'] + ' (' + df['iteration'].astype(str) + ')',
                                 textposition="bottom center"))

        fig.add_trace(go.Scatter(x=df['used_budget'], y=df['predicted_f1_score'],
                                 mode='lines+markers',
                                 name='Predicted',
                                 line_shape='spline',
                                 marker_color='#949494',
                                 line_dash='dash',))

    if len(df_ac) > 0:
        df_ac['iteration'] = df_ac['iteration'].astype(float)
        df_ac = df_ac.sort_values(by=['iteration'], ascending=True)
        df_ac['used_budget'] = df_ac['used_budget'].astype(float)
        df_ac['used_budget'] = 1.0
        df_ac.loc[df_ac['iteration'] == 0, 'used_budget'] = 0
        df_ac['used_budget'] = df_ac['used_budget'].cumsum()
        df_ac = df_ac[df_ac['used_budget'] <= 50]
        df_ac.loc[df_ac['iteration'] == 0, 'feature'] = ''
        df_ac = df_ac.sort_values(by=['used_budget'], ascending=True)
        fig.add_trace(go.Scatter(x=df_ac['used_budget'], y=df_ac['f1_score'],
                                 mode='lines+markers',
                                 name='ActiveClean',
                                 line_shape='spline',
                                 textposition="bottom right"))



    if len(df_cleaned) > 0:
        df_cleaned['used_budget'] = df_cleaned['used_budget'].cumsum()
        df_cleaned = df_cleaned[df_cleaned['used_budget'] <= 50]
        fig.add_trace(go.Scatter(x=df_cleaned['used_budget'], y=df_cleaned['real_f1'],
                                 mode='lines',
                                 marker_color='black',
                                 name='Cleaned',
                                 line_shape='spline',
                                 textposition="bottom right",
                                 line=dict(
                                     dash='dashdot',
                                     width=2
                                 )))


    fig.update_layout(
        xaxis=dict(
            tickmode='array'
        )
    )

    fig = fig.update_layout(showlegend=True,
                            xaxis_title='used budget',
                            yaxis_title='F1',
                            title=f'Performance per cleaning iteration')

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox('Toggle matplotlib visibility', key=pre_pollution_setting_id):

        create_matplotlib_plot(df, df_ac, df_cleaned)


def create_matplotlib_plot(df, df_ac, df_cleaned):
    sns.set_context("paper")
    sns.set(style="whitegrid", font_scale=1.92)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df['used_budget'], df['f1_score'], 'o-', color='#0173B2', label='COMET', linestyle='-', marker='o')

    segments = []
    start = 0
    for i, value in enumerate(df['predicted_f1_score']):
        if value == 0.5:
            if start != i:
                segments.append((start, i))
            start = i + 1
    if start < len(df['predicted_f1_score']):
        segments.append((start, len(df['predicted_f1_score'])))
    first_segment_plotted = False
    for start, end in segments:
        segment = df.iloc[start:end]
        if not first_segment_plotted:
            ax.plot(segment['used_budget'], segment['predicted_f1_score'], 'o-', color='#949494', label='Predicted', linestyle='--',
                    marker='D')
            first_segment_plotted = True
        else:
            ax.plot(segment['used_budget'], segment['predicted_f1_score'], 'o-', color='#949494', linestyle='--',
                    marker='D')
    if len(df_ac) > 0:
        ax.errorbar(df_ac['iteration'], df_ac['f1_score'], fmt='o-',
                    color='#FFC107', ecolor='#FFC107', elinewidth=0.599, label='AC', linestyle='-',
                    marker='v',
                    alpha=0.99)
    else:
        df_ac = pd.DataFrame(columns=['used_budget', 'real_f1', 'feature'])
        ax.plot(df_ac['used_budget'], df_ac['f1_score'], 'o-', color='#FFC107', label='AC',
                linestyle='-', marker='v')
    if len(df_cleaned) > 0:
        ax.plot(df_cleaned['used_budget'], df_cleaned['real_f1'], 'black', label='Cleaned', linestyle='-.', linewidth=2)
    else:
        df_cleaned = pd.DataFrame(columns=['used_budget', 'real_f1'])
        ax.plot(df_cleaned['used_budget'], df_cleaned['real_f1'], 'black', label='Cleaned', linestyle='-.', linewidth=2)

    ax.set_xlabel('Used Budget')
    ax.set_ylabel('F1')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels())
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels)
    st.pyplot(fig)

    buffer = BytesIO()
    plt.savefig(buffer, format="pdf")
    buffer.seek(0)

    pdf_plot = buffer.getvalue()

    def get_pdf_download_link(pdf_plot, download_name):
        """Generates a link to download the PDF plot"""
        b64 = base64.b64encode(pdf_plot).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{download_name}">Download plot as PDF</a>'
        return href

    st.markdown(get_pdf_download_link(pdf_plot, "plot.pdf"), unsafe_allow_html=True)


with open('metadata.json', 'r') as f:
    metadata = load_json(f)

DS_NAME = ['cmc.data', 'TelcoCustomerChurn.csv', 'EEG.arff', 'SouthGermanCredit.csv', 'Airbnb', 'Credit', 'Titanic']
DATABASE_URL = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
DATABASE_ENGINE = create_engine(DATABASE_URL, echo=True)


st.set_page_config(
    page_title='COMET',
    page_icon='✔',
    layout='wide'
)

st.title('COMET evaluation considering the single-error scenario (AC)')

st.sidebar.title('Select configuration')
error_type = st.sidebar.selectbox(
    'Select ML algorithm',
    [SGDClassifierExperimentSVM.__name__, SGDClassifierExperimentLogisticRegression.__name__, SGDClassifierExperimentLinearRegression.__name__])

experiment_str_new = st.sidebar.selectbox(
    'Select modifier',
    [CategoricalShiftModifier.get_classname(), GaussianNoiseModifier.get_classname(), MissingValuesModifier.get_classname(), ScalingModifier.get_classname()])

ds_name = st.sidebar.selectbox(
    'Select ds_name',
    DS_NAME)


def load_and_filter_data(base_name, ds_name, experiment_str, modifier_str, database_url):
    data = load_data_from_db(f'{base_name}_{ds_name}_{experiment_str}_{modifier_str}', database_url)
    if len(data) != 0:
        if 'dataset' in data.columns:
            data = data[
                (data['dataset'] == ds_name) &
                (data['polluter'] == modifier_str) &
                (data['experiment'] == experiment_str)
            ]
    return data

dynamic_greedy_cleaning_schedule = load_and_filter_data(
    'cleaning_results', ds_name, error_type, experiment_str_new, DATABASE_URL)

activeclean_results = load_and_filter_data(
    'activeclean_results', ds_name, error_type, experiment_str_new, DATABASE_URL)

cleaned_data = load_and_filter_data(
    'cleaned_data_results', ds_name, error_type, experiment_str_new, DATABASE_URL)

pre_pollution_settings_train_df = load_data_from_db(f'pre_pollution_settings_{ds_name}_train', DATABASE_URL)
if pre_pollution_settings_train_df.empty:
    pre_pollution_settings_train_df = load_data_from_db(f'pre_pollution_settings_{ds_name}', DATABASE_URL)
pre_pollution_settings_test_df = load_data_from_db(f'pre_pollution_settings_{ds_name}_test', DATABASE_URL)
if pre_pollution_settings_test_df.empty:
    pre_pollution_settings_test_df = load_data_from_db(f'pre_pollution_settings_{ds_name}', DATABASE_URL)
#st.write(pre_pollution_settings_df)
pre_pollution_setting_ids = pre_pollution_settings_train_df['pre_pollution_setting_id'].unique()
for pre_pollution_setting_id in pre_pollution_setting_ids[0:3]:
    if not dynamic_greedy_cleaning_schedule.empty:
        dynamic_greedy_cleaning_schedule_filtered = dynamic_greedy_cleaning_schedule[dynamic_greedy_cleaning_schedule['pre_pollution_setting_id'] == pre_pollution_setting_id]
    else:
        dynamic_greedy_cleaning_schedule_filtered = pd.DataFrame()
    if not activeclean_results.empty:
        activeclean_results_filtered = activeclean_results[activeclean_results['pre_pollution_setting_id'] == pre_pollution_setting_id]
    else:
        activeclean_results_filtered = pd.DataFrame()
    if not cleaned_data.empty and not dynamic_greedy_cleaning_schedule_filtered.empty:
        cleaned_data_filtered = cleaned_data[cleaned_data['pre_pollution_setting_id'] == pre_pollution_setting_id]
        cleaned_data_filtered = pd.concat([cleaned_data_filtered]*len(dynamic_greedy_cleaning_schedule_filtered), ignore_index=True)
        cleaned_data_filtered['used_budget'] = dynamic_greedy_cleaning_schedule_filtered['used_budget'].values
    else:
        cleaned_data_filtered = pd.DataFrame()

    st.subheader(f'Pre-pollution setting {pre_pollution_setting_id}')
    st.write('#### Train')
    st.write(pre_pollution_settings_train_df[pre_pollution_settings_train_df["pre_pollution_setting_id"] == pre_pollution_setting_id].drop('pre_pollution_setting_id', axis=1))
    st.write('#### Test')
    st.write(pre_pollution_settings_test_df[pre_pollution_settings_test_df["pre_pollution_setting_id"] == pre_pollution_setting_id].drop('pre_pollution_setting_id', axis=1))
    plot_performance_per_budget_with_optimal(pre_pollution_setting_id,
                                             dynamic_greedy_cleaning_schedule_filtered,
                                             activeclean_results_filtered, cleaned_data_filtered)
    st.write('---')
