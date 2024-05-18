import joblib
import numpy as np
import pandas as pd

encoder_Marital_status = joblib.load("/workspaces/predict_student/model/encoder_Marital_status.joblib")
encoder_Application_mode = joblib.load("/workspaces/predict_student/model/encoder_Application_mode.joblib")
encoder_Previous_qualification = joblib.load("/workspaces/predict_student/model/encoder_Previous_qualification.joblib")
encoder_Displaced	= joblib.load("/workspaces/predict_student/model/encoder_Displaced.joblib")
encoder_Debtor = joblib.load("/workspaces/predict_student/model/encoder_Debtor.joblib")
encoder_Tuition_fees_up_to_date	= joblib.load("/workspaces/predict_student/model/encoder_Tuition_fees_up_to_date.joblib")
encoder_Gender = joblib.load("/workspaces/predict_student/model/encoder_Gender.joblib")
encoder_Scholarship_holder = joblib.load("/workspaces/predict_student/model/encoder_Scholarship_holder.joblib")
scaler_Age_at_enrollment = joblib.load("/workspaces/predict_student/model/scaler_Age_at_enrollment.joblib")
scaler_Admission_grade = joblib.load("/workspaces/predict_student/model/scaler_Admission_grade.joblib")
scaler_Age_at_enrollment = joblib.load("/workspaces/predict_student/model/scaler_Age_at_enrollment.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_1st_sem_without_evaluations = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_1st_sem_without_evaluations.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_without_evaluations = joblib.load("/workspaces/predict_student/model/scaler_Curricular_units_2nd_sem_without_evaluations.joblib")


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
    
