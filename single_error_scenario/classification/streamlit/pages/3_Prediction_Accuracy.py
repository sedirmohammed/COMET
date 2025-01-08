import streamlit as st
from sqlalchemy import create_engine
from classification.utils.util import load_data_from_db
from classification.experiments import KNeighborsExperiment, DecisionTreeExperiment, \
    MultilayerPerceptronExperiment, SupportVectorMachineExperiment, GradientBoostingExperiment, SGDClassifierExperimentSVM, \
    SGDClassifierExperimentLinearRegression,  SGDClassifierExperimentLogisticRegression
from classification.utils.DatasetModifier import MissingValuesModifier, GaussianNoiseModifier, ScalingModifier, \
    CategoricalShiftModifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64
from sklearn.metrics import mean_absolute_error
from config.definitions import ROOT_DIR

st.set_page_config(
    page_title='COMET',
    page_icon='✔',
    layout='wide'
)

st.title('Prediction accuracy considering the single-srror scenario')

DS_NAMES = ['cmc.data', 'TelcoCustomerChurn.csv', 'EEG.arff', 'SouthGermanCredit.csv', 'Airbnb', 'Credit', 'Titanic']
EXPERIMENTS = [GradientBoostingExperiment.__name__, KNeighborsExperiment.__name__, MultilayerPerceptronExperiment.__name__, SupportVectorMachineExperiment.__name__, SGDClassifierExperimentSVM.__name__, SGDClassifierExperimentLinearRegression.__name__, SGDClassifierExperimentLogisticRegression.__name__]
ERROR_TYPES = [CategoricalShiftModifier.get_classname(), GaussianNoiseModifier.get_classname(), MissingValuesModifier.get_classname(), ScalingModifier.get_classname()]

DATABASE_URL = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
DATABASE_ENGINE = create_engine(DATABASE_URL, echo=True)


mae_results = pd.DataFrame(columns=['Error Type', 'Experiment', 'MAE'])

for error_type in ERROR_TYPES:
    for experiment in EXPERIMENTS:
        all_dfs = []

        for ds_name in DS_NAMES:
            table_name = f'cleaning_results_{ds_name}_{experiment}_{error_type}'
            df = load_data_from_db(table_name, DATABASE_ENGINE)
            if not df.empty:
                df = df[df['predicted_f1_score'] != 0.5]
                all_dfs.append(df)

        combined_df = pd.concat(all_dfs)
        if not combined_df.empty:
            # Calculate MAE
            mae = mean_absolute_error(combined_df['f1_score'], combined_df['predicted_f1_score'])
            temp_df = pd.DataFrame({'Error Type': error_type,
                                    'Experiment': experiment,
                                    'MAE': mae}, index=[0])
            mae_results = pd.concat([mae_results, temp_df])


experiment_labels = {
    'GradientBoostingExperiment': 'GB',
    'KNeighborsExperiment': 'KNN',
    'MultilayerPerceptronExperiment': 'MLP',
    'SupportVectorMachineExperiment': 'SVM',
    'SGDClassifierExperimentSVM': 'AC SVM',
    'SGDClassifierExperimentLinearRegression': 'LIR',
    'SGDClassifierExperimentLogisticRegression': 'LOR'
}

error_type_labels = {
    'CategoricalShiftModifier': 'Categorical\nShift',
    'GaussianNoiseModifier': 'Gaussian\nNoise',
    'MissingValuesModifier': 'Missing\nValues',
    'ScalingModifier': 'Scaling'
}

mae_results['Experiment'] = mae_results['Experiment'].map(experiment_labels)
mae_results['Error Type'] = mae_results['Error Type'].map(error_type_labels)
mae_results = mae_results.reset_index(drop=True)
mae_results['MAE'] = mae_results['MAE'].astype(float)
print(mae_results.to_string())

sns.set_context('paper')

sns.set(style='whitegrid', font_scale=1.85)
colorblind_palette = ['#CC78BC', '#949494', '#ECE133', '#56B4E9', '#81712B', '#004D40', '#89AC17']
fig, ax = plt.subplots(figsize=(11, 7))
sns.barplot(x='Error Type', y='MAE', hue='Experiment', data=mae_results, ax=ax, palette=colorblind_palette)

ax.set_ylabel('MAE')
ax.set_xlabel('Error Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels())
legend = ax.legend(loc='upper left', ncols=4)
legend.set_title(None)
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

st.markdown(get_pdf_download_link(pdf_plot, "plot.pdf"), unsafe_allow_html=True)




