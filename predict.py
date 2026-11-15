"""Serve the heart disease model as a web service.

Run it with:  uv run uvicorn predict:app --host 0.0.0.0 --port 9696
"""

import pickle
from typing import Literal

import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    fill_values, dv, model, threshold = pickle.load(f_in)

feature_names = list(dv.get_feature_names_out())


# the input: one patient, with the same column names and values as the prepared data

class Patient(BaseModel):
    age: int = Field(..., gt=0)
    sex: Literal['m', 'f']
    chest_pain_type: Literal['asy', 'nap', 'ata', 'ta']
    resting_bp: float | None = Field(None, gt=0)    # leave out if not measured
    cholesterol: float | None = Field(None, gt=0)   # leave out if not measured
    fasting_bs: Literal[0, 1]
    resting_ecg: Literal['normal', 'lvh', 'st']
    max_hr: int = Field(..., gt=0)
    exercise_angina: Literal['y', 'n']
    oldpeak: float
    st_slope: Literal['up', 'flat', 'down']


class PredictResponse(BaseModel):
    heart_disease_probability: float
    heart_disease: bool


app = FastAPI(title='heart-disease-prediction')


def predict_single(patient):
    # fill missing values with the training medians saved in model.bin
    for column, value in fill_values.items():
        if patient[column] is None:
            patient[column] = float(value)

    X = dv.transform([patient])
    dX = xgb.DMatrix(X, feature_names=feature_names)
    return float(model.predict(dX)[0])


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(patient: Patient) -> PredictResponse:
    probability = predict_single(patient.model_dump())

    return PredictResponse(
        heart_disease_probability=round(probability, 3),
        heart_disease=probability >= threshold,
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)
