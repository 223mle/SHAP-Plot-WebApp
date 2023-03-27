import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from src.preprocess import Preprocess
from src.shap_plot import ShapPlot

# icon設定
st.set_page_config(page_title='ozro_wepapp',
                    page_icon='clubhouse-icon.png')

st.title('Explanation of GBDT prediction using SHAP')

st.markdown('* * *')
st.markdown('## Please Upload Model and Train data')

upload_model_file = st.file_uploader('Please Upload Model(pickle file) here.', type='pkl')
st.markdown('* * *')
upload_train_file = st.file_uploader('Please Upload data file used to train the model (csv file).', type='csv')
st.markdown('* * *')

@st.cache_data
def load_data(upload_model_file, upload_train_file):
    model = pd.read_pickle(upload_model_file)
    train = pd.read_csv(upload_train_file)
    return model, train


if (upload_model_file is not None) and (upload_train_file is not None):
    model, train = load_data(upload_model_file, upload_train_file)

    st.session_state['model'] = model
    st.session_state['train'] = train


    #unuse_cols = st.multiselect(
    #    '解析不要な列があれば選択してください.',
    #    df.columns
    #)
    #if len(unuse_cols)==len(df.columns):
    #    st.error('解析する列がありません.')

