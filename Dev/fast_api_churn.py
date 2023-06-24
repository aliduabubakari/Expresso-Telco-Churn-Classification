import os
import sys

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from typing import List
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Create an instance of FastAPI
app = FastAPI(debug=True)

# Load the trained models and transformers
num_imputer = joblib.load('numerical_imputer.joblib')
cat_imputer = joblib.load('cat_imputer.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')
model = joblib.load('lr_model_vif_smote.joblib')

original_feature_names = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
                          'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK',
                          'REGION_DAKAR', 'REGION_DIOURBEL', 'REGION_FATICK', 'REGION_KAFFRINE', 'REGION_KAOLACK',
                          'REGION_KEDOUGOU', 'REGION_KOLDA', 'REGION_LOUGA', 'REGION_MATAM', 'REGION_SAINT-LOUIS',
                          'REGION_SEDHIOU', 'REGION_TAMBACOUNDA', 'REGION_THIES', 'REGION_ZIGUINCHOR',
                          'TENURE_Long-term', 'TENURE_Medium-term', 'TENURE_Mid-term', 'TENURE_Short-term',
                          'TENURE_Very short-term', 'TOP_PACK_data', 'TOP_PACK_international', 'TOP_PACK_messaging',
                          'TOP_PACK_other_services', 'TOP_PACK_social_media', 'TOP_PACK_value_added_services',
                          'TOP_PACK_voice']


class InputData(BaseModel):
    MONTANT: float
    FREQUENCE_RECH: float
    REVENUE: float
    ARPU_SEGMENT: float
    FREQUENCE: float
    DATA_VOLUME: float
    ON_NET: float
    ORANGE: float
    TIGO: float
    ZONE1: float
    ZONE2: float
    REGULARITY: float
    FREQ_TOP_PACK: float
    REGION: str
    TENURE: str
    TOP_PACK: str


def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, index=[0])

    cat_columns = ['REGION', 'TENURE', 'TOP_PACK']
    num_columns = [col for col in input_df.columns if col not in cat_columns]

    input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns])
    input_df_imputed_num = num_imputer.transform(input_df[num_columns])

    input_encoded_df = pd.DataFrame(encoder.transform(input_df_imputed_cat).toarray(),
                                    columns=encoder.get_feature_names_out(cat_columns))

    input_df_scaled = scaler.transform(input_df_imputed_num)
    input_scaled_df = pd.DataFrame(input_df_scaled, columns=num_columns)
    final_df = pd.concat([input_encoded_df, input_scaled_df], axis=1)
    final_df = final_df.reindex(columns=original_feature_names, fill_value=0)

    return final_df


def make_prediction(data, model):
    probabilities = model.predict_proba(data)
    churn_labels = ["No Churn" if class_idx == 0 else "Churn" for class_idx in range(len(probabilities[0]))]
    churn_probabilities = probabilities[0]

    # Get the predicted churn label and its probability
    predicted_class_index = np.argmax(churn_probabilities)
    predicted_churn_label = churn_labels[predicted_class_index]
    predicted_probability = churn_probabilities[predicted_class_index]

    # Customize the output message based on the predicted churn label and its probability
    if predicted_churn_label == "Churn":
        output_message = f"⚠️ Customer is likely to churn with a probability of {predicted_probability:.2f}. This indicates a high risk of losing the customer. ⚠️"
    else:
        output_message = f"✅ Customer is not likely to churn with a probability of {predicted_probability:.2f}. This indicates a lower risk of losing the customer. ✅"

    return output_message

@app.get("/")
def read_root():

    info = """
    Welcome to the Expressor Churn Prediction API!. This API provides advanced machine learning predictions for churn. ⚡📊 For more information and to explore the API's capabilities, please visit the documentation: https://abubakari-sepsis-fastapi-prediction-app.hf.space/docs/
    """
    return info.strip()

# Model information endpoint
@app.post('/model-info')
async def model_info():
    model_name = model.__class__.__name__  # get model name
    model_params = model.get_params()  # get model parameters
    model_information = {
        'model info': {
            'model name': model_name,
            'model parameters': model_params
        }
    }
    return model_information  # return model information

@app.post('/predict')
async def predict(input_data: InputData):
    input_features = input_data.dict()
    preprocessed_data = preprocess_input(input_features)
    prediction = make_prediction(preprocessed_data, model)

    return {"prediction": prediction}


@app.post('/batch_predict')
async def predict(input_data: List[InputData]):
    preprocessed_data = []

    for data in input_data:
        input_features = data.dict()
        preprocessed = preprocess_input(input_features)
        preprocessed_data.append(preprocessed)

    predictions = [make_prediction(data, model) for data in preprocessed_data]

    return {"predictions": predictions}
