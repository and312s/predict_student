import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Tab 1'

def switch_tab(tab_name):
    st.session_state.active_tab = tab_name

if st.session_state.active_tab == 'Tab 1':
    gradient_text = """
    <style>
    .gradient-text {
        background: linear-gradient(90deg, purple, blue);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 30px; 
    }
    </style>
    """
    st.markdown(gradient_text, unsafe_allow_html=True)
    
    st.markdown('<p class="gradient-text">AI Insights for Educators: Boost Student Graduation Rates.</p>', unsafe_allow_html=True)
    
    st.caption('Welcome to "AI Insights for Educators"! Our platform is tailored to provide university educators with predictive analytics on student graduation and dropout risks. Use our insights to develop targeted interventions and support your students towards successful academic careers.')

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Next"):
            switch_tab('Tab 2')

elif st.session_state.active_tab == 'Tab 2':
    st.write("Konten Tab 2")
    if st.button("Previous"):
        switch_tab('Tab 1')
    if st.button("Predict"):
        switch_tab('Tab 3')

elif st.session_state.active_tab == 'Tab 3':
    st.write("Konten Tab 3")
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Finish"):
            switch_tab('Tab 1')

