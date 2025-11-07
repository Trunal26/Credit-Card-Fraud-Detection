# api_model_wrapper.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import tensorflow as tf
import uvicorn

app = FastAPI(title="Fraud Detection API")

# load model + scaler
MODEL_PATH = "model/mlp_model.h5"
SCALER_PATH = "model/scaler.joblib"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define the input schema - include all features Time,V1..V28,Amount
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float
    V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float
    V19: float; V20: float; V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float
    V28: float
    Amount: float

@app.get("/")
def root():
    return {"status": "API is up. Post to /predict"}

@app.post("/predict")
def predict(tx: Transaction):
    # convert to dataframe
    data = pd.DataFrame([tx.dict()])
    # scale Time & Amount
    data[["Time","Amount"]] = scaler.transform(data[["Time","Amount"]])
    # predict
    prob = float(model.predict(data.values)[0][0])
    label = int(prob > 0.5)
    return {"probability": prob, "label": label}

if __name__ == "__main__":
    uvicorn.run("api_model_wrapper:app", host="0.0.0.0", port=8000, reload=True)
