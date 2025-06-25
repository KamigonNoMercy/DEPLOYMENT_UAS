from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

with open("best_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS'
]

app = FastAPI()

class ObesityFeatures(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

@app.get("/")
def home():
    return {"message": "Obesity Prediction API is running."}

@app.post("/predict")
def predict(data: ObesityFeatures):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[feature_names]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model.classes_.tolist()

    # Ambil feature importance jika model ada atribut ini
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        importance_dict = dict(zip(feature_names, importances))
    else:
        importance_dict = {}

    return {
        "prediction": pred,
        "probabilities": dict(zip(classes, proba)),
        "feature_importances": importance_dict
    }
