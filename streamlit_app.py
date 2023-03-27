import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from preprocess import Preprocess


# icon設定
st.set_page_config(page_title='ozro_wepapp',
                    page_icon='clubhouse-icon.png')

st.title('Explanation of GBDT prediction using SHAP')

upload_model_file = st.file_uploader('Please Upload Model(pickle file) here.', type='pkl')
upload_train_file = st.file_uploader('Please Upload data file used to train the model (csv file).', type='csv')


if (upload_model_file is not None) and (upload_train_file is not None):
    model = pd.read_pickle(upload_model_file)
    train = pd.read_csv(upload_train_file)

    explainer = shap.TreeExplainer(model, data=train)
    shap_val = explainer.shap_values(train)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values=shap_val,
                    features=train,
                    feature_names=train.columns)
    st.pyplot(fig)

    #unuse_cols = st.multiselect(
    #    '解析不要な列があれば選択してください.',
    #    df.columns
    #)
    #if len(unuse_cols)==len(df.columns):
    #    st.error('解析する列がありません.')

