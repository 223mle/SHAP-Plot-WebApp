import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.shap_plot import ShapPlot


st.title('Individual Visualization')

st.markdown('### This page shows waterfall Plot with SHAP')

if ('model' not in st.session_state) or ('train' not in st.session_state):
    st.error('Please Upload Model and Train Data file')

else:
    st.info('It takes a few seconds to display the plot')
    model = st.session_state['model']
    train = st.session_state['train']

    select_row = st.selectbox(
        'Please select Visualization Data row number',
        list(range(len(train)))
    )

    shap_plot = ShapPlot(model, train)
    fig = shap_plot.waterfall_plot(select_row)
    st.pyplot(fig)
