import joblib
import pandas as pd
from fastapi import FastAPI

from fraud_detection.config import MODEL_PATH, FRAUD_THRESHOLD
from fraud_detection.schemas import Transaction, PredictionResponse

app = FastAPI(title='Fraud Detection API')

model = joblib.load(MODEL_PATH)


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