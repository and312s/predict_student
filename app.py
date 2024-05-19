import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing, encoder_Marital_status, encoder_Scholarship_holder, encoder_Application_mode, encoder_Previous_qualification, encoder_Displaced, encoder_Debtor, encoder_Tuition_fees_up_to_date, encoder_Gender, encoder_Scholarship_holder
from prediction import prediction

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
    
    st.markdown('<p class="gradient-text">AI Insights for Jaya Jaya Institut Educators: Boost Student Graduation Rates.</p>', unsafe_allow_html=True)
    
    st.caption('Welcome to "AI Insights for Educators"! Our platform is tailored to provide Jaya Jaya Institut educators with predictive analytics on student graduation and dropout risks. Use our insights to develop targeted interventions and support your students towards successful academic careers.')

    col1, col2 = st.columns([7, 1])
    with col1:
        st.write(" ")
    with col2:
        if st.button("Next"):
            switch_tab('Tab 2')

if st.session_state.active_tab == 'Tab 2':
    col1, col2, col3, col4, col5= st.columns(5)
    data = pd.DataFrame()

    with col1: 
        Marital_status = st.selectbox(label="Marital status", options=encoder_Marital_status.classes_, index=1)
        data["Marital_status"] = [Marital_status]

    with col2: 
        Gender = st.selectbox(label="Gender", options=encoder_Gender.classes_, index=1)
        data["Gender"] = [Gender]

    with col3: 
        Displaced = st.selectbox(label="Displaced", options=encoder_Displaced.classes_, index=1)
        data["Displaced"] = [Displaced]
    
    with col4: 
        Debtor = st.selectbox(label="Debtor", options=encoder_Debtor.classes_, index=1)
        data["Debtor"] = [Debtor]
    
    with col5:
        Age_at_enrollment = int(st.number_input(label="Age At Enrollment", min_value=17, max_value=70))
        data["Age_at_enrollment"] = [Age_at_enrollment]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: 
        Application_mode = st.selectbox(label="Application mode", options=encoder_Application_mode.classes_, index=1)
        data["Application_mode"] = [Application_mode]
    
    with col2: 
        Previous_qualification = st.selectbox(label="Previous qualification", options=encoder_Previous_qualification.classes_, index=1)
        data["Previous_qualification"] = [Previous_qualification]
    
    with col3: 
        Scholarship_holder = st.selectbox(label="Scholarship holder", options=encoder_Scholarship_holder.classes_, index=1)
        data["Scholarship_holder"] = [Scholarship_holder]
    
    with col4: 
        Tuition_fees_up_to_date = st.selectbox(label="Tuition fees up to date", options=encoder_Tuition_fees_up_to_date.classes_, index=1)
        data["Tuition_fees_up_to_date"] = [Tuition_fees_up_to_date]
    
    with col5: 
        Admission_grade = int(st.number_input(label="Admission_grade", max_value=200))
        data["Admission_grade"] = [Admission_grade]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        Curricular_units_1st_sem_approved = float(st.number_input(label="Curricular units 1st sem approved", max_value=100))
        data["Curricular_units_1st_sem_approved"] = [Curricular_units_1st_sem_approved]
    
    with col2:
        Curricular_units_1st_sem_enrolled = float(st.number_input(label="Curricular units 1st sem enrolled", max_value=100))
        data["Curricular_units_1st_sem_enrolled"] = [Curricular_units_1st_sem_enrolled]
    
    with col3:
        Curricular_units_1st_sem_evaluations = float(st.number_input(label="Curricular units 1st sem evaluation", max_value=100))
        data["Curricular_units_1st_sem_evaluations"] = [Curricular_units_1st_sem_evaluations]
    
    with col4:
        Curricular_units_1st_sem_grade = float(st.number_input(label="Curricular units 1st sem grade", max_value=100))
        data["Curricular_units_1st_sem_grade"] = [Curricular_units_1st_sem_grade]
    
    with col5:
        Curricular_units_1st_sem_without_evaluations = float(st.number_input(label="Curricular units 1st sem without evaluations", max_value=100))
        data["Curricular_units_1st_sem_without_evaluations"] = [Curricular_units_1st_sem_without_evaluations]


    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        Curricular_units_2nd_sem_approved = float(st.number_input(label="Curricular units 2nd sem approved", max_value=100))
        data["Curricular_units_2nd_sem_approved"] = [Curricular_units_2nd_sem_approved]
    
    with col2:
        Curricular_units_2nd_sem_enrolled = float(st.number_input(label="Curricular units 2nd sem enrolled", max_value=100))
        data["Curricular_units_2nd_sem_enrolled"] = [Curricular_units_2nd_sem_enrolled]
    
    with col3:
        Curricular_units_2nd_sem_evaluations = float(st.number_input(label="Curricular units 2nd sem evaluation", max_value=100))
        data["Curricular_units_2nd_sem_evaluations"] = [Curricular_units_2nd_sem_evaluations]
    
    with col4:
        Curricular_units_2nd_sem_grade = float(st.number_input(label="Curricular units 2nd sem grade", max_value=100))
        data["Curricular_units_2nd_sem_grade"] = [Curricular_units_2nd_sem_grade]
    
    with col5:
        Curricular_units_2nd_sem_without_evaluations = float(st.number_input(label="Curricular units 2nd sem without evaluations", max_value=100))
        data["Curricular_units_2nd_sem_without_evaluations"] = [Curricular_units_2nd_sem_without_evaluations]


    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Previous"):
            switch_tab('Tab 1')

    with col3:
        if st.button("Predict"):
            new_data = data_preprocessing(data=data)
            st.session_state.new_data = new_data  
            switch_tab('Tab 3')
            st.experimental_rerun() 

if st.session_state.active_tab == 'Tab 3':
    if 'new_data' in st.session_state:
        new_data = st.session_state.new_data
        with st.expander("View The Preprocessed Data"):
            st.dataframe(data=new_data, width=800, height=400)
        st.write("Result: {}".format(prediction(new_data)))


    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Finish"):
            switch_tab('Tab 1')

