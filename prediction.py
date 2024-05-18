import joblib

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
