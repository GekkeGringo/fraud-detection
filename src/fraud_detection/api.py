import joblib
import pandas as pd
from fastapi import FastAPI

from fraud_detection.config import FRAUD_THRESHOLD, MODEL_PATH
from fraud_detection.schemas import PredictionResponse, Transaction

app = FastAPI(title='Fraud Detection API')

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get('/health')
def health_check():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictionResponse)
def predict(transaction: Transaction):
    input_df = pd.DataFrame([transaction.model_dump()])

    proba = model.predict_proba(input_df)[:, 1][0]
    is_fraud = bool(proba >= FRAUD_THRESHOLD)

    return PredictionResponse(
        fraud_probability=float(proba),
        is_fraud=is_fraud,
        threshold_used=FRAUD_THRESHOLD,
    )