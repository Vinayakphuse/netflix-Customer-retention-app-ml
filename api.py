from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import json
import shap
import uvicorn
import numpy as np

app = FastAPI(title="Customer Churn Prediction API", description="ML Pipeline Endpoint")

# Initialize variables with defaults
model = None
explainer = None
feature_names = []
encoder_mapping = {}

# Load artifacts
try:
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('encoder_mapping.json', 'r') as f:
        encoder_mapping = json.load(f)
except FileNotFoundError:
    print("WARNING: Model artifacts not found. Please run train.py first.")

class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(...)
    subscription_type: str = Field(...)
    watch_hours: float = Field(..., ge=0.0)
    last_login_days: int = Field(..., ge=0)
    region: str = Field(...)
    device: str = Field(...)
    monthly_fee: float = Field(..., ge=0.0)
    payment_method: str = Field(...)
    number_of_profiles: int = Field(..., ge=1)
    avg_watch_time_per_day: float = Field(..., ge=0.0)
    favorite_genre: str = Field(...)

class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    shap_values: dict
    base_value: float

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please train the model first.")

    # Prepare input data
    input_dict = {
        'age': customer.age,
        'watch_hours': customer.watch_hours,
        'last_login_days': customer.last_login_days,
        'monthly_fee': customer.monthly_fee,
        'number_of_profiles': customer.number_of_profiles,
        'avg_watch_time_per_day': customer.avg_watch_time_per_day,
    }
    
    # Encode categorical features
    categorical_cols = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']
    for col in categorical_cols:
        val = getattr(customer, col)
        if col not in encoder_mapping or val not in encoder_mapping[col]:
            raise HTTPException(status_code=400, detail=f"Invalid value '{val}' for {col}. Allowed: {list(encoder_mapping.get(col, {}).keys())}")
        input_dict[col] = encoder_mapping[col][val]
    
    df = pd.DataFrame([input_dict])
    # Ensure correct column order
    df = df[feature_names]
    
    # Predict
    prob = model.predict_proba(df)[0, 1]
    prediction = int(prob > 0.5)
    
    # SHAP Explainability
    shap_val = explainer(df)
    
    # shap_val.values is an array of shape (1, num_features)
    # We want to map each feature to its SHAP impact
    feature_impacts = {feature: float(val) for feature, val in zip(feature_names, shap_val.values[0])}
    
    return PredictionResponse(
        churn_probability=float(prob),
        will_churn=bool(prediction),
        shap_values=feature_impacts,
        base_value=float(shap_val.base_values[0])
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
