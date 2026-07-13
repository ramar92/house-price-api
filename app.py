from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message":"House Price Prediction API"}

@app.post("/predict")
def predict(area:float, bedrooms:int, age:int):

    sample = pd.DataFrame({
        "Area":[area],
        "Bedrooms":[bedrooms],
        "Age":[age]
    })

    prediction = model.predict(sample)

    return {"Predicted Price":float(prediction[0])}