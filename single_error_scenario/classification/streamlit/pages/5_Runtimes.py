import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from config.definitions import ROOT_DIR

st.set_page_config(
    page_title='COMET',
    page_icon='✔',
    layout='wide'
)

st.title('Runtime of COMET, considering the single-error scenario')


DATABASE_URL = f'sqlite:///{ROOT_DIR}/db/RESULTS.db'
DATABASE_ENGINE = create_engine(DATABASE_URL, echo=True)

def load_data_from_csv(file_path):
    """Loads data from a CSV file"""
    return pd.read_csv(file_path)

# Load runtime data without averaging
runtimes_df = load_data_from_csv(f'{ROOT_DIR}/paper/runtimes.csv')
runtimes_df.drop('dataset', axis=1, inplace=True)
runtimes_df['ml_algorithm'] = runtimes_df['ml_algorithm'].replace({'Gradient': 'GB', 'AC_SVM': 'AC\nSVM'})
runtimes_df['error_type'] = runtimes_df['error_type'].replace({'Categorical Shift': 'CS', 'Gaussian Noise': 'GN', 'Missing Values': 'MV', 'Scaling': 'S'})

# Set Seaborn style
sns.set(style='whitegrid', font_scale=1.85)
colorblind_palette = ['#CC78BC', '#949494', '#ECE133', '#56B4E9']
error_type_order = ['CS', 'GN', 'MV', 'S']
ml_algorithm_order = ['GB', 'KNN', 'MLP', 'SVM', 'AC\nSVM', 'LIR', 'LOR']


fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(x='ml_algorithm', y='runtime', hue='error_type', data=runtimes_df, ax=ax, palette=colorblind_palette, hue_order=error_type_order, order=ml_algorithm_order)
ax.set_ylabel('Runtime (s)')
ax.set_xlabel('ML Algorithm')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Adjust rotation for better label visibility
legend = ax.legend(loc='upper right', ncols=2)
plt.tight_layout()
st.pyplot(fig)

buffer = BytesIO()
plt.savefig(buffer, format="pdf")
buffer.seek(0)
pdf_plot = buffer.getvalue()

st.write('## Median Runtimes')
median_runtimes = runtimes_df.groupby(['ml_algorithm', 'error_type']).median().reset_index()
st.write(median_runtimes)

st.write('## Mean Runtimes')
mean_runtimes = runtimes_df.groupby(['ml_algorithm', 'error_type']).mean().reset_index()
st.write(mean_runtimes)

def get_pdf_download_link(pdf_plot, download_name):
    """Generates a link to download the PDF plot"""
    b64 = base64.b64encode(pdf_plot).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{download_name}">Download plot as PDF</a>'
    return href

st.markdown(get_pdf_download_link(pdf_plot, "plot.pdf"), unsafe_allow_html=True)
