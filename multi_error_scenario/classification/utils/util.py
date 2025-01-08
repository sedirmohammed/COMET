import pandas as pd
from pandas import read_sql
import itertools
from sqlalchemy import exc, MetaData, Table
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import sessionmaker
import os
import subprocess
from config.definitions import ROOT_DIR
import warnings
import streamlit as st
from io import BytesIO
import base64
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_pre_pollution_df(ds_name, error_type, database_engine):
    pre_pollution_df = load_data_from_db(f'pre_pollution_{ds_name}_{error_type.__name__}', database_engine)
    if pre_pollution_df.empty:
        print(f'pre_pollution_df is empty. Create pre_pollution with current settings ({error_type.__name__}, {ds_name})')
        ds_path = os.path.join(ROOT_DIR, 'data', ds_name)
        cmd = ['python3', '../pre_pollution.py', f'--error_type={error_type.__name__}',  f'--dataset={ds_path}']
        subprocess.Popen(cmd).wait()
        pre_pollution_df = load_data_from_db(f'pre_pollution_{ds_name}_{error_type.__name__}', database_engine)

    if pre_pollution_df.empty:
        warnings.warn(f'pre_pollution_df is empty. Something went wrong during the creation of '
                      f'pre_pollution_{ds_name}_{error_type.__name__} table in the database. '
                      f'Check whether paths are correct and the input data exists in the defined folder.', UserWarning)
    else:
        pre_pollution_df['pollution_level'] = pre_pollution_df['pollution_level'].astype(float)
        pre_pollution_df['seed'] = pre_pollution_df['seed'].astype(int)
    return pre_pollution_df


def create_combinations_from_top_k_features(top_k_features):
    for modifier_str in top_k_features.keys():
        current_top_k_features = top_k_features[modifier_str].copy()
        combinations = []
        for L in range(1, len(current_top_k_features) + 1):
            if L > 1:
                break
            for subset in itertools.combinations(current_top_k_features, L):
                combinations.append(list(subset))
        top_k_features[modifier_str] = combinations
    return top_k_features


def load_data_from_db(table_name, database_engine):
    try:
        df = read_sql(
            f'SELECT * FROM "{table_name}"',
            con=database_engine
        )
    except exc.OperationalError:
        print(f'Table with name: {table_name} does not exist')
        return pd.DataFrame()
    return df


def write_time_measurement_to_db(table_name, ds_name, experiment_name, error_type, measured_time, pre_pollution_setting_id, database_engine):
    time_measurement = {'dataset': ds_name, 'experiment': experiment_name, 'error_type': error_type, 'measured_time': measured_time, 'pre_pollution_setting_id': pre_pollution_setting_id}

    time_measurements_df = load_data_from_db(table_name, database_engine)
    if len(time_measurements_df) == 0:
        time_measurements_df = pd.DataFrame(time_measurement, index=[0])
    else:
        time_measurements_df = time_measurements_df[time_measurements_df['dataset'] != ds_name &
                                                    time_measurements_df['experiment'] != experiment_name &
                                                    time_measurements_df['error_type'] != error_type &
                                                    time_measurements_df['pre_pollution_setting_id'] != pre_pollution_setting_id]
        time_measurements_df = time_measurements_df.append(time_measurement, ignore_index=True)
    time_measurements_df.to_sql(table_name, con=database_engine, if_exists='replace', index=False)


def get_pre_pollution_settings(ds_name, database_engine, selected_pre_pollution_setting_ids=None):
    pre_pollution_settings_df = load_data_from_db(f'pre_pollution_settings_{ds_name}', database_engine)
    if pre_pollution_settings_df.empty:
        pre_pollution_settings_df_train = load_data_from_db(f'pre_pollution_settings_{ds_name}_train', database_engine)
        pre_pollution_settings_df_train = pre_pollution_settings_df_train.sort_values(by=['pre_pollution_setting_id'], ascending=True)
        pre_pollution_settings_train = pre_pollution_settings_df_train.to_dict('records')

        pre_pollution_settings_df_test = load_data_from_db(f'pre_pollution_settings_{ds_name}_test', database_engine)
        pre_pollution_settings_df_test = pre_pollution_settings_df_test.sort_values(by=['pre_pollution_setting_id'], ascending=True)
        pre_pollution_settings_test = pre_pollution_settings_df_test.to_dict('records')

        # zip both list of dicts to one list of dicts
        pre_pollution_settings = [{'train': ps1, 'test': ps2} for ps1, ps2 in zip(pre_pollution_settings_train, pre_pollution_settings_test)]
    else:
        pre_pollution_settings_df = pre_pollution_settings_df.sort_values(by=['pre_pollution_setting_id'], ascending=True)
        pre_pollution_settings = pre_pollution_settings_df.to_dict('records')
        pre_pollution_settings = [{'train': ps.copy(), 'test': ps.copy()} for ps in pre_pollution_settings]

    temp_results = []
    for pre_pollution_setting in pre_pollution_settings:
        pre_pollution_setting_id = pre_pollution_setting['train']['pre_pollution_setting_id']

        temp_results.append({'pre_pollution_setting_id': pre_pollution_setting_id, 'train': pre_pollution_setting['train'], 'test': pre_pollution_setting['test']})
    pre_pollution_settings = temp_results
    results = []
    if selected_pre_pollution_setting_ids is not None:
        for pre_pollution_setting in pre_pollution_settings:
            if pre_pollution_setting['pre_pollution_setting_id'] in selected_pre_pollution_setting_ids:
                results.append(pre_pollution_setting)
    else:
        results = pre_pollution_settings
    return results


def drop_table(table_name, database_engine):
    inspector = Inspector.from_engine(database_engine)
    if table_name in inspector.get_table_names():
        metadata = MetaData()    # Define the table
        selected_table = Table(table_name, metadata, autoload=True, autoload_with=database_engine)
        selected_table.drop(database_engine)


def delete_entries_from_table(table_name, database_engine, pre_pollution_setting_ids):

    metadata = MetaData(bind=database_engine)
    from sqlalchemy.engine import reflection
    inspector = reflection.Inspector.from_engine(database_engine)

    # Check if the table exists in the database before reflecting
    if not inspector.has_table(table_name):
        print(f'Table with name: {table_name} does not exist')
        return

    try:
        selected_table = Table(table_name, metadata, autoload=True, autoload_with=database_engine)
    except NoSuchTableError as e:
        print(f'Error reflecting table: {e}')
        return

    # Create a new session
    Session = sessionmaker(bind=database_engine)
    session = Session()

    # Perform the query and deletion
    session.query(selected_table).filter(
        selected_table.c.pre_pollution_setting_id.in_(pre_pollution_setting_ids)
    ).delete(synchronize_session='fetch')
    session.commit()


def impute_missing_rows(group):
    group = group.sort_values(by="used_budget")
    result = pd.DataFrame(columns=group.columns)
    expected_budget = 1

    for _, row in group.iterrows():
        previous_row = result.iloc[-1] if len(result) > 0 else row
        while row['used_budget'] > expected_budget:
            new_row = row.copy()
            new_row['used_budget'] = expected_budget
            new_row['feature'] = previous_row['feature'] + "_imputed"
            new_row['real_f1'] = previous_row['real_f1']
            result = result.append(new_row, ignore_index=True)
            expected_budget += 1
        result = result.append(row, ignore_index=True)
        expected_budget = row['used_budget'] + 1

    while max(group['used_budget']) >= expected_budget:
        new_row = group.iloc[-1].copy()
        new_row['used_budget'] = expected_budget
        new_row['feature'] += "_imputed"
        result = result.append(new_row, ignore_index=True)
        expected_budget += 1
    # remove rows that start with the string <BUFFER> in the feature column
    result = result[~result['feature'].str.startswith('<BUFFER>')]
    return result


def load_and_prepare_data(table_name, database_url):
    df = load_data_from_db(f'{table_name}', database_url)
    if len(df) == 0:
        return pd.DataFrame()
    df['iteration'] = df['iteration'].astype(float)
    df = df.sort_values(by=['iteration'], ascending=True)
    df['used_budget'] = df['used_budget'].fillna(0)
    df['used_budget'] = df['used_budget'].astype(int)
    df['used_budget'] = df.groupby(['pre_pollution_setting_id'])['used_budget'].cumsum()
    if 'f1_score' in df.columns:
        df.rename({'f1_score': 'real_f1'}, axis=1, inplace=True)

    all_results = []
    for setting_id, group in df.groupby('pre_pollution_setting_id'):
        processed_group = impute_missing_rows(group)
        mean_scores = processed_group.groupby('used_budget')['real_f1'].mean().reset_index(name='real_f1')
        mean_iterations = processed_group.groupby('used_budget')['iteration'].mean().reset_index()
        concat_features = processed_group.groupby('used_budget')['feature'].apply(lambda x: ', '.join(x)).reset_index()
        aggregated_data = pd.merge(mean_scores, concat_features, on='used_budget')
        aggregated_data = pd.merge(aggregated_data, mean_iterations, on='used_budget')
        aggregated_data['pre_pollution_setting_id'] = setting_id
        all_results.append(aggregated_data)
    df = pd.concat(all_results).reset_index(drop=True)
    return df

def plot_performance_agg(agg_results_df, title=''):
    fig = go.Figure()

    agg_results_df['color'] = np.where(agg_results_df['mean_diff'] < 0, 'rgb(213,94,0)', 'rgb(2,158,115)')
    agg_results_df = agg_results_df.sort_values(by=['used_budget'], ascending=True)
    fig.add_trace(go.Bar(x=agg_results_df['used_budget'], y=agg_results_df['mean_diff'],
                         base=0.0,
                         name='Difference in F1',
                         marker_color=agg_results_df['color'],
                         error_y=dict(
                             type='data',  # value of error bar given in data coordinates
                             array=agg_results_df['std_diff'],
                             visible=True)
                         ))

    fig.update_layout(
        xaxis=dict(
            tickmode='array'
        )
    )

    fig = fig.update_layout(showlegend=False,
                            xaxis_title='used budget',
                            yaxis_title='F1 difference',
                            title=f'{title}')

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox('Toggle matplotlib visibility', key=title):

        agg_results_df['color_hex'] = np.where(agg_results_df['mean_diff'] < 0, '#D55E00', '#0173B2')
        agg_results_df['used_budget'] = agg_results_df['used_budget'].astype(int)

        sns.set_context("paper")
        sns.set(style="whitegrid", font_scale=1.)
        fig, ax = plt.subplots(figsize=(3, 2))

        sns.barplot(x='used_budget', y='mean_diff', data=agg_results_df,
                    ci=None, capsize=0.2, ax=ax,
                    yerr=agg_results_df['std_diff'], palette=agg_results_df['color_hex'], width=1., linewidth=0.0, dodge=False)

        ax.set_xlabel('Used Budget')
        ax.set_ylabel('F1 Difference')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ticks = np.arange(19, len(agg_results_df), 20)
        ax.set_xticks(ticks)
        ax.set_xticklabels((ticks + 1).astype(str))

        custom_y_ticks = [-0.005, 0.0, 0.01, 0.02, 0.025]
        ax.set_yticks(custom_y_ticks)

        plt.tight_layout()
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

        st.markdown(get_pdf_download_link(pdf_plot, f"{title.replace(' ', '_')}.pdf"), unsafe_allow_html=True)
