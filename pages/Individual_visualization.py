import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.shap_plot import ShapPlot


st.title('Individual Visualization')

st.markdown('### This page shows waterfall Plot with SHAP')

model = st.session_state['model']
train = st.session_state['train']

select_row = st.selectbox(
    'Please select Visualization Data row number',
    list(range(len(train)))
)

shap_plot = ShapPlot(model, train)
fig = shap_plot.waterfall_plot(select_row)
st.pyplot(fig)
