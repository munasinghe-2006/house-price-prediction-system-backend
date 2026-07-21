import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pickle
from pydantic import BaseModel
import numpy as np

load_dotenv()

frontend_url = os.getenv("FRONTEND_URL")


class HousePricePredictionInput(BaseModel):
    total_sqft: float
    bath: int
    balcony: int
    bhk: int
    area_type: str
    location: str


app = FastAPI(
    title="FastAPI Template",
    description="This is a sample FastAPI template.",
    version="1.0.0",)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent / "models"/"bengaluru_house_price_linear_regression_model.pickle"

try:
    with open(MODEL_PATH, "rb") as file:
        model_package = pickle.load(file)

        model = model_package["model"]
        feature_columns = model_package["feature_columns"]
 
except FileNotFoundError:
    raise RuntimeError(f"The pickle file was not found at the specified path: {MODEL_PATH}")

except KeyError:
    raise RuntimeError("The pickle file does not contain the expected keys: 'model' and 'feature_columns'.")

except Exception as e:
    raise RuntimeError(f"An unexpected error occurred: {e}")

NUMERICAL_TYPE_FEATURES = ["total_sqft", "bath", "balcony", "bhk"]

AREA_TYPE_FEATURES = ["Built-up  Area", "Carpet  Area", "Plot  Area", "Super built-up  Area"]

LOCATION_TYPE_FEATURES = [
    feature for feature in feature_columns if feature not in NUMERICAL_TYPE_FEATURES and feature not in AREA_TYPE_FEATURES
]

def create_input_row(data: HousePricePredictionInput):
    input_row = np.zeros(len(feature_columns))

    numerical_features = {
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony,
        "bhk": data.bhk
    }

    for feature, value in numerical_features.items():
        feature_index = feature_columns.index(feature)
        input_row[feature_index] = value

    if data.area_type not in AREA_TYPE_FEATURES:
        raise HTTPException(status_code=400, detail=f"Invalid area_type. Must be one of: {AREA_TYPE_FEATURES}")
    
    area_type_index = feature_columns.index(data.area_type)
    input_row[area_type_index] = 1

    if data.location not in LOCATION_TYPE_FEATURES:
        raise HTTPException(status_code=400, detail=f"Invalid location. Must be one of: {LOCATION_TYPE_FEATURES}")
    
    location_index = feature_columns.index(data.location)
    input_row[location_index] = 1

    return input_row
              

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Template!"}

@app.get("/health")
def read_health():
    return {"status": "ok"}


@app.get("/model-info")
def read_model_info():
    return {
        "model_type": type(model).__name__,
        "no_of_feature_columns": len(feature_columns),
        "sample_feature_columns": feature_columns[:10]
    }

@app.get("/options")
def read_options():
    return {
        "area_types": AREA_TYPE_FEATURES,
        "location_types": LOCATION_TYPE_FEATURES
    }

@app.post("/predict")
def predict(data: HousePricePredictionInput):
    input_row = create_input_row(data)
    prediction_result = model.predict([input_row])[0]

    return{
        "prediction": round(prediction_result, 2),
           "input_data": {
               "total_sqft": data.total_sqft,
               "bath": data.bath,
               "balcony": data.balcony,
               "bhk": data.bhk,
               "area_type": data.area_type,
               "location": data.location
           }
           }
    