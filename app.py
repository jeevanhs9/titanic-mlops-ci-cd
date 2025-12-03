from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import yaml

app = FastAPI(title="Titanic Survival Prediction API")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model():
    config = load_config()
    latest_model_path = config["paths"]["latest_model"]
    model = joblib.load(latest_model_path)
    return model


# Request schema for prediction
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.get("/")
def root():
    return {"message": "Titanic model is running!"}


@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert Pydantic model to DataFrame row
    import pandas as pd

    data = pd.DataFrame([passenger.dict()])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    return {
        "survived": int(pred),
        "survival_probability": float(prob)
    }
