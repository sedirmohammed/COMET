import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from json import load as load_json
from classification.utils.util import load_data_from_db
from classification.experiments import DecisionTreeExperiment, \
    MultilayerPerceptronExperiment, SupportVectorMachineExperiment, GradientBoostingExperiment, KNeighborsExperiment
from classification.utils.DatasetModifier import MissingValuesModifier, GaussianNoiseModifier, ScalingModifier, \
    CategoricalShiftModifier
import plotly.graph_objects as go
from config.definitions import ROOT_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def plot_performance_per_budget_with_optimal(pre_pollution_setting_id, df, df_fi, df_optimal, df_fi_static, df_c_random, df_cleaned, df_comet_light):
    try:
        original_f1_score = df[df['iteration'] == 0]['f1_score'].values[0]
    except:
        original_f1_score = 0.5
    fig = go.Figure()

    if len(df) > 0:
        #df = df[~df['feature'].str.startswith('<BUFFER>')]
        df = df.sort_values(by=['iteration'], ascending=True)
        df['used_budget'] = df['used_budget'].astype(int)
        df['used_budget'] = df['used_budget'].cumsum()
        df = df[df['used_budget'] <= 50]
        df = df.sort_values(by=['used_budget'], ascending=True)
        # sort by used budget and then by iteration
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

    if len(df_comet_light) > 0:
        df_comet_light = df_comet_light.sort_values(by=['iteration'], ascending=True)
        df_comet_light['used_budget'] = df_comet_light['used_budget'].astype(int)
        df_comet_light['used_budget'] = df_comet_light['used_budget'].cumsum()
        df_comet_light = df_comet_light[df_comet_light['used_budget'] <= 50]
        df_comet_light = df_comet_light.sort_values(by=['used_budget'], ascending=True)
        # sort by used budget and then by iteration
        df_comet_light = df_comet_light.sort_values(by=['iteration', 'used_budget'], ascending=True)
        fig.add_trace(go.Scatter(x=df_comet_light['used_budget'], y=df_comet_light['f1_score'],
                                 mode='lines+markers',
                                 marker_color='#b20101',
                                 name='COMET Light',
                                 line_shape='spline',
                                 text=df_comet_light['feature'] + ' (' + df_comet_light['iteration'].astype(str) + ')',
                                 textposition="bottom center"))

    if len(df_fi) > 0:
        df_fi['iteration'] = df_fi['iteration'].astype(float)
        df_fi.loc[df_fi['iteration'] == 0, 'used_budget'] = 0
        df_fi = df_fi.sort_values(by=['iteration'], ascending=True)
        df_fi['used_budget'] = df_fi['used_budget'].astype(float)
        df_fi['used_budget'] = df_fi['used_budget'].cumsum()
        df_fi = df_fi[df_fi['used_budget'] <= 50]
        df_fi['feature'] = df_fi['feature'] + ' (' + df_fi['used_budget'].astype(str) + ')'
        df_fi.loc[df_fi['iteration'] == 0, 'feature'] = ''
        df_fi = df_fi.sort_values(by=['used_budget'], ascending=True)
        fig.add_trace(go.Scatter(x=df_fi['used_budget'], y=df_fi['f1_score'],
                                 mode='lines+markers',
                                 name='feature importance f1',
                                 line_shape='spline',
                                 text=df_fi['feature'],
                                 textposition="bottom right"))

    if len(df_fi_static) > 0:
        df_fi_static['iteration'] = df_fi_static['iteration'].astype(float)
        df_fi_static.loc[df_fi_static['iteration'] == 0, 'used_budget'] = 0
        df_fi_static = df_fi_static.sort_values(by=['iteration'], ascending=True)
        df_fi_static['used_budget'] = df_fi_static['used_budget'].astype(float)
        df_fi_static['used_budget'] = df_fi_static['used_budget'].cumsum()
        df_fi_static = df_fi_static[df_fi_static['used_budget'] <= 50]
        df_fi_static['feature'] = df_fi_static['feature'] + ' (' + df_fi_static['used_budget'].astype(str) + ')'
        df_fi_static.loc[df_fi_static['iteration'] == 0, 'feature'] = ''
        df_fi_static = df_fi_static.sort_values(by=['used_budget'], ascending=True)
        fig.add_trace(go.Scatter(x=df_fi_static['used_budget'], y=df_fi_static['f1_score'],
                                 mode='lines+markers',
                                 marker_color='#029E73',
                                 name='Feature importance',
                                 line_shape='spline',
                                 text=df_fi_static['feature'],
                                 textposition="bottom right"))

    if len(df_optimal) > 0:
        new_row = pd.DataFrame({
            'iteration': [0],
            'feature_combination': ['initial f1'],
            'real_f1': [original_f1_score],
            'predicted_poly_reg_f1': [original_f1_score]
        })
        df_optimal = pd.concat([df_optimal, new_row], ignore_index=True)
        df_optimal.loc[df_optimal['iteration'] == 0, 'used_budget'] = 0
        df_optimal = df_optimal.sort_values(by=['iteration'], ascending=True)
        df_optimal['used_budget'] = df_optimal['used_budget'].astype(int)
        df_optimal['used_budget'] = df_optimal['used_budget'].cumsum()
        df_optimal = df_optimal[df_optimal['used_budget'] <= 50]
        df_optimal['feature_combination'] = df_optimal['feature_combination'] + ' (' + df_optimal['used_budget'].astype(str) + ')'
        df_optimal.loc[df_optimal['iteration'] == 0, 'feature'] = ''
        df_optimal = df_optimal.sort_values(by=['used_budget'], ascending=True)
        fig.add_trace(go.Scatter(x=df_optimal['used_budget'], y=df_optimal['real_f1'],
                                 mode='lines+markers',
                                 marker_color='#CC78BC',
                                 name='Oracle',
                                 line_shape='spline',
                                 text=df_optimal['feature'],
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


    if len(df_c_random) > 0:
        new_row = pd.DataFrame({
            'iteration': [0],
            'feature_combination': ['initial f1'],
            'real_f1': [original_f1_score],
            'predicted_poly_reg_f1': [original_f1_score]
        })
        df_c_random = pd.concat([df_c_random, new_row], ignore_index=True)
        df_c_random.loc[df_c_random['iteration'] == 0, 'used_budget'] = 0
        df_c_random = df_c_random.sort_values(by=['iteration'], ascending=True)
        df_c_random['used_budget'] = df_c_random['used_budget'].astype(int)
        df_c_random = df_c_random.groupby('iteration').head(5).reset_index(drop=True)
        df_c_random = df_c_random[df_c_random['iteration'] <= 50]
        df_c_random['feature_combination'] = df_c_random['feature_combination'] + ' (' + df_c_random['used_budget'].astype(str) + ')'
        df_c_random.loc[df_c_random['iteration'] == 0, 'feature'] = ''
        df_c_random['std_diff'] = 0.0
        df_c_random['mean_diff'] = 0.0

        df_c_random = df_c_random.groupby('iteration').agg(
            feature=pd.NamedAgg(column='feature', aggfunc=lambda x: ', '.join(x)),
            real_f1=pd.NamedAgg(column='real_f1', aggfunc='mean'),
            std=pd.NamedAgg(column='real_f1', aggfunc='std')
        ).reset_index()
        # replace na values in std with 0.0
        df_c_random['std'] = df_c_random['std'].fillna(0.0)
        df_c_random = df_c_random.sort_values(by=['iteration'], ascending=True)
        fig.add_trace(go.Scatter(x=df_c_random['iteration'], y=df_c_random['real_f1'],
                                 mode='lines+markers',
                                 marker_color='#DE8F05',
                                 name='Completely random',
                                 line_shape='spline',
                                 text=df_c_random['feature'],
                                 textposition="bottom right",
                                 error_y=dict(
                                     type='data',  # value of error bar given in data coordinates
                                     array=df_c_random['std'],
                                     visible=True)
                                 ))

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

        create_matplotlib_plot(df, df_c_random, df_fi_static, df_optimal, df_cleaned, df_comet_light)


def create_matplotlib_plot(df, df_c_random, df_fi_static, df_optimal, df_cleaned, df_comet_light):
    sns.set_context("paper")
    # Set the style to your preference
    sns.set(style="whitegrid", font_scale=1.92)
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot the 'real_f1' data
    ax.plot(df['used_budget'], df['f1_score'], 'o-', color='#0173B2', label='COMET', linestyle='-',
            marker='o')
    if len(df_optimal) > 0:
        # Plot the 'real_f1' data
        ax.plot(df_optimal['used_budget'], df_optimal['real_f1'], 'o-', color='#CC78BC', label='Oracle', linestyle='-', marker='x')
    else:
        df_optimal = pd.DataFrame(columns=['used_budget', 'real_f1', 'feature'])
        ax.plot(df_optimal['used_budget'], df_optimal['real_f1'], 'o-', color='#CC78BC', label='Oracle', linestyle='-', marker='x')
    ax.plot(df_fi_static['used_budget'], df_fi_static['f1_score'], 'o-', color='#029E73', label='FIR', linestyle='-', marker='*')
    ax.plot(df_comet_light['used_budget'], df_comet_light['f1_score'], 'o-', color='#b20101', label='CL', linestyle='-', marker='^')

    # Plot the 'predicted_poly_reg_f1' data
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
    if len(df_c_random) > 0:
        ax.errorbar(df_c_random['iteration'], df_c_random['real_f1'], yerr=df_c_random['std'], fmt='o-',
                    color='#DE8F05', ecolor='#DE8F05', elinewidth=0.599, label='RR', linestyle='-',
                    marker='v',
                    alpha=0.99)
    else:
        df_c_random = pd.DataFrame(columns=['used_budget', 'real_f1', 'feature'])
        ax.plot(df_c_random['used_budget'], df_c_random['real_f1'], 'o-', color='#DE8F05', label='RR',
                linestyle='-', marker='v')
    if len(df_cleaned) > 0:
        ax.plot(df_cleaned['used_budget'], df_cleaned['real_f1'], 'black', label='Cleaned', linestyle='-.', linewidth=2)
    else:
        df_cleaned = pd.DataFrame(columns=['used_budget', 'real_f1'])
        ax.plot(df_cleaned['used_budget'], df_cleaned['real_f1'], 'black', label='Cleaned', linestyle='-.', linewidth=2)

    # Adding labels and title
    ax.set_xlabel('Used Budget')
    ax.set_ylabel('F1')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels())
    # plt.title('Performance per cleaning iteration')
    handles, labels = plt.gca().get_legend_handles_labels()
    # Modify the order of handles and labels here (e.g., reversing them)
    handles = [handles[i] for i in [0, 4, 2, 6, 3, 1, 5]]
    labels = [labels[i] for i in [0, 4, 2, 6, 3, 1, 5]]
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

st.title('COMET evaluation considering the single-error scenario')

st.sidebar.title('Select configuration')
error_type = st.sidebar.selectbox(
    'Select ML algorithm',
    [GradientBoostingExperiment.__name__, KNeighborsExperiment.__name__, MultilayerPerceptronExperiment.__name__, SupportVectorMachineExperiment.__name__])

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

comet_light_cleaning_schedule = load_and_filter_data(
    'comet_light_cleaning_results', ds_name, error_type, experiment_str_new, DATABASE_URL)

dynamic_greedy_cleaning_schedule_optimal = load_and_filter_data(
    'cleaning_schedule_optimal', ds_name, error_type, experiment_str_new, DATABASE_URL)

feature_importance_greedy_cleaning_schedule = load_and_filter_data(
    'cleaning_schedule_features_importance', ds_name, error_type, experiment_str_new, DATABASE_URL)
# Resetting the dataframe as per your original code
feature_importance_greedy_cleaning_schedule = pd.DataFrame()

static_feature_importance_greedy_cleaning_schedule = load_and_filter_data(
    'cleaning_schedule_static_features_importance', ds_name, error_type, experiment_str_new, DATABASE_URL)

completely_random_cleaning_schedule = load_and_filter_data(
    'cleaning_schedule_completely_random', ds_name, error_type, experiment_str_new, DATABASE_URL)

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
    if not dynamic_greedy_cleaning_schedule_optimal.empty:
        dynamic_greedy_cleaning_schedule_optimal_filtered = dynamic_greedy_cleaning_schedule_optimal[dynamic_greedy_cleaning_schedule_optimal['pre_pollution_setting_id'] == pre_pollution_setting_id]
    else:
        dynamic_greedy_cleaning_schedule_optimal_filtered = pd.DataFrame()
    if not comet_light_cleaning_schedule.empty:
        comet_light_cleaning_schedule_filtered = comet_light_cleaning_schedule[comet_light_cleaning_schedule['pre_pollution_setting_id'] == pre_pollution_setting_id]
    else:
        comet_light_cleaning_schedule_filtered = pd.DataFrame()
    if not feature_importance_greedy_cleaning_schedule.empty:
        feature_importance_greedy_cleaning_schedule_filtered = feature_importance_greedy_cleaning_schedule[feature_importance_greedy_cleaning_schedule['pre_pollution_setting_id'] == pre_pollution_setting_id]
    else:
        feature_importance_greedy_cleaning_schedule_filtered = feature_importance_greedy_cleaning_schedule
    if not static_feature_importance_greedy_cleaning_schedule.empty:
        static_feature_importance_greedy_cleaning_schedule_filtered = static_feature_importance_greedy_cleaning_schedule[static_feature_importance_greedy_cleaning_schedule['pre_pollution_setting_id'] == pre_pollution_setting_id]
        static_feature_importance_greedy_cleaning_schedule_filtered['used_budget'] = static_feature_importance_greedy_cleaning_schedule_filtered['used_budget'].fillna('0').astype(int)
    else:
        static_feature_importance_greedy_cleaning_schedule_filtered = static_feature_importance_greedy_cleaning_schedule
    if not completely_random_cleaning_schedule.empty:
        completely_random_cleaning_schedule_filtered = completely_random_cleaning_schedule[completely_random_cleaning_schedule['pre_pollution_setting_id'] == pre_pollution_setting_id]
    else:
        completely_random_cleaning_schedule_filtered = completely_random_cleaning_schedule
    if not cleaned_data.empty and not static_feature_importance_greedy_cleaning_schedule_filtered.empty:
        cleaned_data_filtered = cleaned_data[cleaned_data['pre_pollution_setting_id'] == pre_pollution_setting_id]
        cleaned_data_filtered = pd.concat([cleaned_data_filtered]*len(static_feature_importance_greedy_cleaning_schedule_filtered), ignore_index=True)
        cleaned_data_filtered['used_budget'] = static_feature_importance_greedy_cleaning_schedule_filtered['used_budget'].values
    else:
        cleaned_data_filtered = pd.DataFrame()

    st.subheader(f'Pre-pollution setting {pre_pollution_setting_id}')
    st.write('#### Train')
    st.write(pre_pollution_settings_train_df[pre_pollution_settings_train_df["pre_pollution_setting_id"] == pre_pollution_setting_id].drop('pre_pollution_setting_id', axis=1))
    st.write('#### Test')
    st.write(pre_pollution_settings_test_df[pre_pollution_settings_test_df["pre_pollution_setting_id"] == pre_pollution_setting_id].drop('pre_pollution_setting_id', axis=1))
    plot_performance_per_budget_with_optimal(pre_pollution_setting_id,
                                             dynamic_greedy_cleaning_schedule_filtered,
                                             feature_importance_greedy_cleaning_schedule_filtered,
                                             dynamic_greedy_cleaning_schedule_optimal_filtered,
                                             static_feature_importance_greedy_cleaning_schedule_filtered,
                                             completely_random_cleaning_schedule_filtered,
                                             cleaned_data_filtered,
                                             comet_light_cleaning_schedule_filtered)
    st.write('---')
