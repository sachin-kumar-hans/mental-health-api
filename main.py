from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize app
app = FastAPI(title="AI-based Mental Health Prediction")

# Load model once at startup
model = joblib.load("mentalh_model.pkl")
vectorizer = joblib.load("vectorizr.pkl")

# Request schema
class TextInput(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "Mental Health API is running 🚀"}

# Prediction endpoint
@app.post("/predict")
def predict(data: TextInput):
    cleaned = data.text.lower()
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)

    return {"prediction": result[0]}