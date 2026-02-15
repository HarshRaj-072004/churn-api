from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated,Literal
import joblib
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse

model= joblib.load('final_churn_model.pkl')
features=joblib.load('feature_names.pkl')
scaler=joblib.load('scaler.pkl')    

app=FastAPI()

class UserInput(BaseModel):
    creditScore: Annotated[float, Field(...,ge=300, le=850,description="Credit score must be between 300 and 850")]
    geography: Annotated[str, Field(..., description="Geography must be one of 'France', 'Spain', or 'Germany'")]
    Age: Annotated[int, Field(..., ge=18, le=100, description="Age must be between 18 and 100")]
    Gender:Annotated[Literal["Male", "Female"], Field(..., description="Gender must be either Male or Female")]
    Tenure: Annotated[int, Field(..., ge=0, le=10, description="Tenure must be between 0 and 10")]
    Balance: Annotated[float, Field(..., ge=0, description="Balance must be non-negative")]
    NumOfProducts: Annotated[int, Field(..., ge=1, le=4, description="Number of products must be between 1 and 4")]
    HasCrCard: Annotated[int, Field(..., ge=0, le=1, description="HasCrCard must be either 0 or 1 depending on whether the customer has a credit card or not")]
    IsActiveMember: Annotated[int, Field(..., ge=0, le=1, description="Enter 1 if the customer is an active member, otherwise enter 0")]
    EstimatedSalary: Annotated[float, Field(..., ge=0, description="Estimated salary must be non-negative")]


@app.post("/predict")
def predict_churn(data: UserInput):
    
    input_data={
        'CreditScore': data.creditScore,
        'Geography': data.geography,
        'Age': data.Age,
        "Gender": data.Gender,  
        'Tenure': data.Tenure,
        'Balance': data.Balance,
        'NumOfProducts': data.NumOfProducts,
        'HasCrCard': data.HasCrCard,
        'IsActiveMember': data.IsActiveMember,
        'EstimatedSalary': data.EstimatedSalary

    }

    df=pd.DataFrame([input_data])
    df=pd.get_dummies(df, columns=['Geography', 'Gender'])

    for feature in features:
        if feature not in df.columns:
            df[feature]=0

    df=df[features]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]


    outcome = "Churn" if pred == 1 else "No Churn" 

    return JSONResponse(status_code=200, content={"churn_prediction": outcome, "churn_probability(in percentage)": float(prob)*100})
