from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="ML Model Comparison API",
    version="1.0"
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"
model = joblib.load(MODEL_PATH)


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

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "survival_probability": round(float(probability), 4)
    }
