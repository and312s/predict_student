import streamlit as st
import pandas as pd
import numpy as np
import joblib

encoder_Marital_status = joblib.load("./model/encoder_Marital_status.joblib")
encoder_Application_mode = joblib.load("./model/encoder_Application_mode.joblib")
encoder_Previous_qualification = joblib.load("./model/encoder_Previous_qualification.joblib")
encoder_Displaced	= joblib.load("./model/encoder_Displaced.joblib")
encoder_Debtor = joblib.load("./model/encoder_Debtor.joblib")
encoder_Tuition_fees_up_to_date	= joblib.load("./model/encoder_Tuition_fees_up_to_date.joblib")
encoder_Gender = joblib.load("./model/encoder_Gender.joblib")
encoder_Scholarship_holder = joblib.load("./model/encoder_Scholarship_holder.joblib")
scaler_Age_at_enrollment = joblib.load("./model/scaler_Age_at_enrollment.joblib")
scaler_Admission_grade = joblib.load("./model/scaler_Admission_grade.joblib")
scaler_Age_at_enrollment = joblib.load("./model/scaler_Age_at_enrollment.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("./model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("./model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("./model/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("./model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_1st_sem_without_evaluations = joblib.load("./model/scaler_Curricular_units_1st_sem_without_evaluations.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("./model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("./model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("./model/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("./model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_without_evaluations = joblib.load("./model/scaler_Curricular_units_2nd_sem_without_evaluations.joblib")


def data_preprocessing(data):
    """Preprocessing data
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()
    
    df["Marital_status"] = encoder_Marital_status.transform(data["Marital_status"])
    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])
    df["Previous_qualification"] = encoder_Previous_qualification.transform(data["Previous_qualification"])
    df["Displaced"] = encoder_Displaced.transform(data["Displaced"])
    df["Debtor"] = encoder_Debtor.transform(data["Debtor"])
    df["Tuition_fees_up_to_date"] = encoder_Tuition_fees_up_to_date.transform(data["Tuition_fees_up_to_date"])
    df["Gender"] = encoder_Gender.transform(data["Gender"])
    df["Scholarship_holder"] = encoder_Scholarship_holder.transform(data["Scholarship_holder"])
    df["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1,1))[0]
    df["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data["Curricular_units_1st_sem_enrolled"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_evaluations"] = scaler_Curricular_units_1st_sem_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_evaluations"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_without_evaluations"] = scaler_Curricular_units_1st_sem_without_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_without_evaluations"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data["Curricular_units_2nd_sem_enrolled"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_evaluations"] = scaler_Curricular_units_2nd_sem_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_evaluations"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_without_evaluations"] = scaler_Curricular_units_2nd_sem_without_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_without_evaluations"]).reshape(-1,1))[0]
    
    return df
    

model = joblib.load("model/lr_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

expected_features = [
    'Marital_status', 'Application_mode', 'Previous_qualification', 'Admission_grade', 
    'Displaced', 'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 
    'Age_at_enrollment', 'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations', 
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 
    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_enrolled', 
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved', 
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
]

def prediction(data):
    """Making prediction

    Args:
        data (Pandas DataFrame): Dataframe that contains all the preprocessed data

    Returns:
        str: Prediction result (Graduate, Dropout)
    """
    # Reorder the columns of data to match expected_features
    data = data[expected_features]

    # Make the prediction
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result

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
            st.rerun() 

if st.session_state.active_tab == 'Tab 3':
    if 'new_data' in st.session_state:
        new_data = st.session_state.new_data
        with st.expander("View The Preprocessed Data"):
            st.dataframe(data=new_data, width=3200, height=10)
        st.write("Result: {}".format(prediction(new_data)))


    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Finish"):
            switch_tab('Tab 1')

