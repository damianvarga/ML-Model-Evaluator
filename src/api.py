from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import subprocess
import sys

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
_model = None


def _ensure_model():
    global _model
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        print("Model not found. Training...")
        subprocess.run([sys.executable, "main.py"], check=True)
    _model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")


app = FastAPI(
    title="ML Model Comparison API",
    version="1.0",
    on_startup=[_ensure_model],
)


def _get_model():
    global _model
    return _model


class Passenger(BaseModel):
    age: int
    sex: str
    fare: float
    class_name: str


@app.get("/")
def root():
    return {"message": "ML Model API running"}


@app.post("/predict")
def predict(passenger: Passenger):

    df = pd.DataFrame([{
        "age": passenger.age,
        "sex": passenger.sex,
        "fare": passenger.fare,
        "class": passenger.class_name
    }])

    m = _get_model()
    prediction = m.predict(df)[0]

    probability = m.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "survival_probability": round(float(probability), 4)
    }
