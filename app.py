import streamlit as st
import pandas as pd
import numpy as np
import joblib

# CSS untuk warna gradasi
gradient_text = """
<style>
.gradient-text {
    background: linear-gradient(90deg, purple, blue);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
"""

# Menampilkan CSS di Streamlit
st.markdown(gradient_text, unsafe_allow_html=True)

# Menampilkan teks dengan warna gradasi
st.markdown('<p class="gradient-text">AI Insights for Educators: Boost Student Graduation Rates.</p>', unsafe_allow_html=True)

st.subheader('Welcome to "AI Insights for Educators"! Our platform is tailored to provide university educators with predictive analytics on student graduation and dropout risks. Use our insights to develop targeted interventions and support your students towards successful academic careers.')
