import joblib

model = joblib_load("model/lr_model.joblib")
result_target = joblib_load("model/encoder_target.joblib")

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Graduate, Dropout)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result
