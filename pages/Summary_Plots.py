import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.shap_plot import ShapPlot

st.title('Summary Plot with SHAP')



st.markdown('### This page shows Summary Plot with SHAP')
st.markdown('[SHAP Documentation(SHAP Summary Plot)](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/NHANES%20I%20Survival%20Model.html?highlight=summary%20plot#SHAP-Summary-Plot)')


if ('model' not in st.session_state) or ('train' not in st.session_state):
    st.error('Please Upload Model and Train Data file')

else:
    st.info('It takes a few seconds to display the plot')
    model = st.session_state['model']
    train = st.session_state['train']


    st.markdown('### SHAP Value(impact on model output)')

    shap = ShapPlot(model, train)
    summary_plot = shap.summary_plot()
    st.pyplot(summary_plot)

    st.markdown('* * *')
    st.markdown('### SHAP Value(average impact on model output)')

    summary_plot_bar = shap.summary_plot_bar()
    st.pyplot(summary_plot_bar)
