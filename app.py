from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel

app = FastAPI()

class Car(BaseModel):
    Year:int
    Present_Price: float
    Kms_Driven:int
    Owner:int
    Fuel_Type_Petrol:int
    Fuel_Type_Diesel:int
    Seller_Type_Individual:int
    Transmission_Mannual:int
path = "random_forest_regression_model.pkl"
with open(path, 'rb') as f:
    model = pickle.load(f)

@app.post("/predict_price")
async def home(item: Car):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)[0]
    result = round(yhat, 5)
    return {"prediction": result}